/*
 * Copyright (C) 2018  Christian Berger
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "cluon-complete.hpp"
#include "opendlv-standard-message-set.hpp"

#include <opencv2/core/types.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>

cv::RNG rng(12345);

int32_t main(int32_t argc, char **argv) {
    int32_t retCode{1};
    auto commandlineArguments = cluon::getCommandlineArguments(argc, argv);
    if ( (0 == commandlineArguments.count("cid")) ||
         (0 == commandlineArguments.count("name")) ||
         (0 == commandlineArguments.count("width")) ||
         (0 == commandlineArguments.count("height")) ) {
        std::cerr << argv[0] << " attaches to a shared memory area containing an ARGB image." << std::endl;
        std::cerr << "Usage:   " << argv[0] << " --cid=<OD4 session> --name=<name of shared memory area> [--verbose]" << std::endl;
        std::cerr << "         --cid:    CID of the OD4Session to send and receive messages" << std::endl;
        std::cerr << "         --name:   name of the shared memory area to attach" << std::endl;
        std::cerr << "         --width:  width of the frame" << std::endl;
        std::cerr << "         --height: height of the frame" << std::endl;
        std::cerr << "Example: " << argv[0] << " --cid=112 --name=img.argb --width=640 --height=480 --verbose" << std::endl;
    }
    else {
        const std::string NAME{commandlineArguments["name"]};
        const uint32_t WIDTH{static_cast<uint32_t>(std::stoi(commandlineArguments["width"]))};
        const uint32_t HEIGHT{static_cast<uint32_t>(std::stoi(commandlineArguments["height"]))};
        const bool VERBOSE{commandlineArguments.count("verbose") != 0};

	const uint32_t MAX_CONTOUR_SIZE = (WIDTH/6)*(HEIGHT/2);
	const uint32_t MIN_CONTOUR_SIZE = (WIDTH/60)*(HEIGHT/60);

        // Attach to the shared memory.
        std::unique_ptr<cluon::SharedMemory> sharedMemory{new cluon::SharedMemory{NAME}};
        if (sharedMemory && sharedMemory->valid()) {
            std::clog << argv[0] << ": Attached to shared memory '" << sharedMemory->name() << " (" << sharedMemory->size() << " bytes)." << std::endl;

            // Interface to a running OpenDaVINCI session; here, you can send and receive messages.
            cluon::OD4Session od4{static_cast<uint16_t>(std::stoi(commandlineArguments["cid"]))};

            // Handler to receive distance readings (realized as C++ lambda).
            std::mutex distancesMutex;
            float front{0};
            float rear{0};
            float left{0};
            float right{0};
            auto onDistance = [&distancesMutex, &front, &rear, &left, &right](cluon::data::Envelope &&env){
                auto senderStamp = env.senderStamp();
                // Now, we unpack the cluon::data::Envelope to get the desired DistanceReading.
                opendlv::proxy::DistanceReading dr = cluon::extractMessage<opendlv::proxy::DistanceReading>(std::move(env));

                // Store distance readings.
                std::lock_guard<std::mutex> lck(distancesMutex);
                switch (senderStamp) {
                    case 0: front = dr.distance(); break;
                    case 2: rear = dr.distance(); break;
                    case 1: left = dr.distance(); break;
                    case 3: right = dr.distance(); break;
                }
            };
            // Finally, we register our lambda for the message identifier for opendlv::proxy::DistanceReading.
            od4.dataTrigger(opendlv::proxy::DistanceReading::ID(), onDistance);

	    std::mutex steerMutex;
	    float steer{0};
	    auto onSteer = [&steerMutex, &steer](cluon::data::Envelope &&env) {
		opendlv::proxy::GroundSteeringRequest sr = cluon::extractMessage<opendlv::proxy::GroundSteeringRequest>(std::move(env));

		std::lock_guard<std::mutex> lck(steerMutex);
		steer = sr.groundSteering();
	    };
	    od4.dataTrigger(opendlv::proxy::GroundSteeringRequest::ID(), onSteer);


            // Endless loop; end the program by pressing Ctrl-C.
            while (od4.isRunning()) {
                cv::Mat img;

                // Wait for a notification of a new frame.
                sharedMemory->wait();

                // Lock the shared memory.
                sharedMemory->lock();
                {
                    // Copy image into cvMat structure.
                    // Be aware of that any code between lock/unlock is blocking
                    // the camera to provide the next frame. Thus, any
                    // computationally heavy algorithms should be placed outside
                    // lock/unlock
                    cv::Mat wrapped(HEIGHT, WIDTH, CV_8UC4, sharedMemory->data());
                    img = wrapped.clone();
                }
                sharedMemory->unlock();

                // TODO: Do something with the frame.

                // Invert colors
                //cv::bitwise_not(img, img);

                // Draw a red rectangle
                //cv::rectangle(img, cv::Point(50, 50), cv::Point(100, 100), cv::Scalar(0,0,255));

		// Crop the top half of the image to remove noises
		cv::Mat cropped_img;
		cv::Rect crop_region(0, HEIGHT/2, WIDTH, HEIGHT/2);
		img = img(crop_region);

		// -----------------------------FILTER BLUE CONES--------------------------------------//
		// Filter the image by color to identify blue cones and yellow cones
		cv::Mat hsv;
		cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

		cv::Scalar hsvBLow(110, 50, 50);
		cv::Scalar hsvBHi(130, 255, 255);

		cv::Mat blueCones;
		cv::inRange(hsv, hsvBLow, hsvBHi, blueCones);

		// Dilate the image
		cv::Mat dilateBlue;
		uint32_t iterations{5};
		cv::dilate(blueCones, dilateBlue, cv::Mat(), cv::Point(-1,-1), iterations, 1, 1);

		// Find contours in the image
		std::vector<std::vector<cv::Point>> contoursBlue;
		cv::findContours( dilateBlue, contoursBlue, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE );

		std::vector<std::vector<cv::Point>> contours_poly_blue(contoursBlue.size());
		std::vector<cv::Rect> boundRectBlue(contoursBlue.size());
		std::vector<cv::Point2f> anchorsBlue(contoursBlue.size());
		std::vector<uint32_t> areasBlue(contoursBlue.size());

		// Filter the contours by minimum and maximum size
		for (size_t i = 0; i < contoursBlue.size(); i++) {
		    cv::approxPolyDP(contoursBlue[i], contours_poly_blue[i], 3, true);
		    boundRectBlue[i] = cv::boundingRect( contours_poly_blue[i] );
		    areasBlue[i] = boundRectBlue[i].height*boundRectBlue[i].width;
		    anchorsBlue[i] = cv::Point2f(boundRectBlue[i].x+boundRectBlue[i].width/2, boundRectBlue[i].y+boundRectBlue[i].height);
		}

		// -----------------------------FILTER YELLOW CONES--------------------------------------//
		cv::Scalar hsvYLow(20, 100, 100);
		cv::Scalar hsvYHi(40, 255, 255);

		cv::Mat yellowCones;
		cv::inRange(hsv, hsvYLow, hsvYHi, yellowCones);

		cv::Mat dilateYellow;
		cv::dilate(yellowCones, dilateYellow, cv::Mat(), cv::Point(-1,-1), iterations, 1, 1);

		std::vector<std::vector<cv::Point>> contoursYellow;
		cv::findContours( dilateYellow, contoursYellow, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE );

		std::vector<std::vector<cv::Point>> contours_poly_yellow(contoursYellow.size());
		std::vector<cv::Rect> boundRectYellow(contoursYellow.size());
		std::vector<cv::Point2f> anchorsYellow(contoursYellow.size());
		std::vector<uint32_t> areasYellow(contoursYellow.size());

		for (size_t i = 0; i < contoursYellow.size(); i++) {
		    cv::approxPolyDP(contoursYellow[i], contours_poly_yellow[i], 3, true);
		    boundRectYellow[i] = cv::boundingRect( contours_poly_yellow[i] );
		    areasYellow[i] = boundRectYellow[i].height*boundRectYellow[i].width;
		    anchorsYellow[i] = cv::Point2f(boundRectYellow[i].x+boundRectYellow[i].width/2, boundRectYellow[i].y+boundRectYellow[i].height);
		}

		// -----------------------------FILTER END--------------------------------------//

		// -----------------------------DRAW CONTOURS-----------------------------------//

		uint32_t max_area_blueCone = 100;

		for( size_t i = 0; i< contoursBlue.size(); i++ )
		{
		    if ((unsigned)boundRectBlue[i].x < WIDTH/4 || ((unsigned)boundRectBlue[i].x < WIDTH/2 && (unsigned)boundRectBlue[i].y > HEIGHT/3)) {
			continue;
		    }

		    if (areasBlue[i] > MAX_CONTOUR_SIZE || areasBlue[i] < MIN_CONTOUR_SIZE) {
			continue;
		    }

		    if (max_area_blueCone==100) {
			max_area_blueCone = i;
		    }else {
			if(areasBlue[i] > areasBlue[max_area_blueCone]) {
			    max_area_blueCone = i;
			}
		    }

		    cv::Scalar blue = cv::Scalar( 255, 0, 0 );
		    cv::rectangle( img, boundRectBlue[i].tl(), boundRectBlue[i].br(), blue, 2 );
		    cv::circle(img, anchorsBlue[i], 10, blue, CV_FILLED, 8, 0);
		}

		uint32_t max_area_yellowCone = 100;

		for( size_t i = 0; i< contoursYellow.size(); i++ )
		{

		    if ((unsigned)boundRectYellow[i].x > WIDTH*3/4) {
			continue;
		    }

		    if (areasYellow[i] > MAX_CONTOUR_SIZE|| areasYellow[i] < MIN_CONTOUR_SIZE) {
			continue;
		    }

		    if (max_area_blueCone==100) {
			max_area_yellowCone = i;
		    }else {
			if(areasYellow[i] > areasYellow[max_area_yellowCone]) {
			    max_area_yellowCone = i;
			}
		    }

		    cv::Scalar yellow = cv::Scalar( 0, 255, 255);
		    cv::rectangle( img, boundRectYellow[i].tl(), boundRectYellow[i].br(), yellow, 2 );
		    cv::circle(img, anchorsYellow[i], 10, yellow, CV_FILLED, 8, 0);
		}

		cv::Point2f aimPoint;
		if(max_area_blueCone != 100 && max_area_yellowCone !=100) {
		    aimPoint.x = (anchorsBlue[max_area_blueCone].x + anchorsYellow[max_area_yellowCone].x)/2;
		    aimPoint.y = (anchorsBlue[max_area_blueCone].y + anchorsYellow[max_area_yellowCone].y)/2;
		} else if (max_area_blueCone != 100) {
		    // TODO: turn left
		    aimPoint.x = WIDTH/2 - 60;
		    aimPoint.y = HEIGHT/2 - 100;
		} else if (max_area_yellowCone != 100) {
		    // TODO: turn right
		    aimPoint.x = WIDTH/2 + 60;
		    aimPoint.y = HEIGHT/2 - 100;
		}

		cv::Scalar red = cv::Scalar( 0, 0, 255);
		cv::circle(img, aimPoint, 10, red, CV_FILLED, 8, 0);

		cv::line(img,cv::Point2f(WIDTH/2, HEIGHT/2),aimPoint,red,5);

		cv::Scalar green = cv::Scalar( 0, 128, 0);
		float angle = steer * 180 / 3.141592;
		cv::Point2f baseline;
		baseline.x = WIDTH/2 - 100*tan(angle);
		baseline.y = HEIGHT/2 - 100;
		cv::line(img,cv::Point2f(WIDTH/2, HEIGHT/2),baseline,green,5);

		// -----------------------------DRAW END--------------------------------------//

                // Display image.
                if (VERBOSE) {
                    cv::imshow(sharedMemory->name().c_str(), img);
                    cv::waitKey(1);
                }

                ////////////////////////////////////////////////////////////////
                // Do something with the distance readings if wanted.
                {
                    std::lock_guard<std::mutex> lck(distancesMutex);
                    std::cout << "front = " << front << ", "
                              << "rear = " << rear << ", "
                              << "left = " << left << ", "
                              << "right = " << right << "." << std::endl;
                }

                ////////////////////////////////////////////////////////////////
                // Example for creating and sending a message to other microservices; can
                // be removed when not needed.
                opendlv::proxy::AngleReading ar;
                ar.angle(123.45f);
                od4.send(ar);

                ////////////////////////////////////////////////////////////////
                // Steering and acceleration/decelration.
                //
                // Uncomment the following lines to steer; range: +38deg (left) .. -38deg (right).
                // Value groundSteeringRequest.groundSteering must be given in radians (DEG/180. * PI).
                //opendlv::proxy::GroundSteeringRequest gsr;
                //gsr.groundSteering(0);
                //od4.send(gsr);

                // Uncomment the following lines to accelerate/decelerate; range: +0.25 (forward) .. -1.0 (backwards).
                // Be careful!
                //opendlv::proxy::PedalPositionRequest ppr;
                //ppr.position(0);
                //od4.send(ppr);
            }
        }
        retCode = 0;
    }
    return retCode;
}

