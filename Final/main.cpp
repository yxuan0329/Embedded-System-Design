#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/select.h>
#include <termios.h>

#include <linux/fb.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/ioctl.h>
#include <opencv2/videoio.hpp>

#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <time.h> // for timer
#include <ctime>


using namespace std;
using namespace cv;

struct termios orig_termios;
struct framebuffer_info get_framebuffer_info ( const char *framebuffer_device_path );
void detectAndDisplay(cv::Mat frame);
void parameterControl(unsigned char key);
//void showCurrTime(time_t startTime, unsigned char key);


cv::CascadeClassifier face_cascade_official, face_cascade_ching, face_cascade_xuan, eyes_cascade, nose_cascade, mouth_cascade;
std::string file_path_official, file_path_ching, file_path_xuan, file_path_eye;
bool shouldDetect = false;
time_t startTime;

void reset_terminal_mode()
{
    tcsetattr(0, TCSANOW, &orig_termios);
}
void set_conio_terminal_mode()
{
    struct termios new_termios;

    /* take two copies - one for now, one for later */
    tcgetattr(0, &orig_termios);
    memcpy(&new_termios, &orig_termios, sizeof(new_termios));

    /* register cleanup handler, and set the new terminal mode */
    atexit(reset_terminal_mode);
    cfmakeraw(&new_termios);
    tcsetattr(0, TCSANOW, &new_termios);
}
int kbhit()
{
    struct timeval tv = { 0L, 0L };
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(0, &fds);
    return select(1, &fds, NULL, NULL, &tv);
}

int getch()
{
    int r;
    unsigned char c;
    if ((r = read(0, &c, sizeof(c))) < 0) {
        return r;
    } else {
        return c;
    }
}

struct framebuffer_info
{
    uint32_t bits_per_pixel;    // depth of framebuffer
    uint32_t xres_virtual;      // how many pixel in a row in virtual screen
    uint32_t yres_virtual;
};

int main ( int argc, const char *argv[] )
{
    const int frame_rate = 10;

    cv::Mat frame;
    cv::Size2f frame_size;

    cv::VideoCapture camera(2);
    framebuffer_info fb_info = get_framebuffer_info("/dev/fb0");

    // open the framebuffer device
    std::ofstream ofs("/dev/fb0");

    if( !camera.isOpened())
    {
        cerr << "Could not open video device." << std::endl;
        return 1;
    }


    // set property of the frame
    int fb_width = fb_info.xres_virtual;
    int fb_height = fb_info.yres_virtual;
    int fb_depth = fb_info.bits_per_pixel;
    // camera.set(cv::CAP_PROP_FRAME_WIDTH, fb_width);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, fb_height);


    int frame_width= camera.get(CV_CAP_PROP_FRAME_WIDTH); // get frame width, height
    int frame_height= camera.get(CV_CAP_PROP_FRAME_HEIGHT);

    file_path_official = "./haarcascades/haarcascade_frontalface_alt2.xml";
    file_path_ching = "./cascade.xml";
    file_path_xuan = "./cascade_xuan.xml";
    if(!face_cascade_official.load(file_path_official)) {
        cout << "Error loading face cascade(official)!\n";
        return -1;
    }
    if(!face_cascade_ching.load(file_path_ching)) {
        cout << "Error loading face cascade(ching)!\n";
        return -1;
    }
    if(!face_cascade_xuan.load(file_path_xuan)) {
        cout << "Error loading face cascade(xuan)!\n";
        return -1;
    }
    file_path_eye = "./haarcascades/haarcascade_eye_tree_eyeglasses.xml";
    if( !eyes_cascade.load( file_path_eye ) )
    {
        cout << "--(!)Error loading eyes cascade\n";
        return -1;
    };

    unsigned char key = ' ';
    while (key!='q')
    {
        // get video frame from stream
        while (!kbhit()) {
          bool ret = camera.read(frame); // cap >> frame;
          if (!ret){
              std::cerr << "Cannot read frame!" << std::endl;
          }

          if (shouldDetect){
            detectAndDisplay(frame);
          }

          // get size of the video frame
          frame_size = frame.size();


          cv::Mat new_frame;
          cvtColor(frame, new_frame, cv::COLOR_BGR2BGR565);

          // output the video frame to framebufer row by row
          for ( int y = 0; y < frame_size.height; y++ ) {
              ofs.seekp(y * fb_width * 2.0f);
              ofs.write(reinterpret_cast<char*>(new_frame.ptr(y)), frame_size.width * 2);
          }
        }
        key = getch();
        parameterControl(key);
    }

    camera.release ();
    return 0;

}

struct framebuffer_info get_framebuffer_info ( const char *framebuffer_device_path )
{
    struct framebuffer_info fb_info;        // Used to return the required attrs.
    struct fb_var_screeninfo screen_info;   // Used to get attributes of the device from OS kernel.

    // open deive with linux system call "open( )"
    int fbfd = 0;
    fbfd = open(framebuffer_device_path, O_RDWR);  // SELF
    if (!fbfd) {                                // SELF
        printf("Error! cannot open framebuffer device!\n");
        return fb_info;
    }
    printf("The framebuffer was opened successfully!\n");

    // get attributes of the framebuffer device through linux system call "ioctl()"
    if (ioctl(fbfd, FBIOGET_VSCREENINFO, &screen_info))
    {
        printf("Error! can't get screen_info information!\n");
        return fb_info;
    }
    printf("%dx%d, %dbpp\n", screen_info.xres, screen_info.yres, screen_info.bits_per_pixel);

    // put the required attributes in variable "fb_info" you found with "ioctl() and return it."
    fb_info.xres_virtual = screen_info.xres_virtual;    // 8
    fb_info.yres_virtual = screen_info.yres_virtual;
    fb_info.bits_per_pixel = screen_info.bits_per_pixel;  // 16
    return fb_info;
};


void detectAndDisplay(cv::Mat frame){
    std::vector<cv::Rect> faces_general, faces_ching, faces_xuan;

    cv::Mat gray_frame;
    // turn frame to grayscale first
    cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
    equalizeHist(gray_frame, gray_frame);


    int unknownID = 0;
    // detect human face
    face_cascade_official.detectMultiScale(gray_frame, faces_general, 0, cv::Size(200,200), cv::Size(350,350));
    //printf("faces_general.size = %d\n", faces_general.size());
    for (size_t i = 0; i < faces_general.size(); i++){
        cv::Point center(faces_general[i].x + faces_general[i].width/2, faces_general[i].y + faces_general[i].height/2);
        rectangle(frame, cv::Point(faces_general[i].x, faces_general[i].y),Point(faces_general[i].x + faces_general[i].width, faces_general[i].y + faces_general[i].height), cv::Scalar(255, 0, 255), 4);

        // rectangle(frame, cv::Point(faces_general[i].x - 30, faces_general[i].y - 30),Point(faces_general[i].x + faces_general[i].width + 60, faces_general[i].y + faces_general[i].height + 60), cv::Scalar(255, 0, 255), 4);
        int rectOffset = 30;
        faces_general[i].x -= rectOffset;
        faces_general[i].y -= rectOffset;
        faces_general[i].x = faces_general[i].x > 0 ? faces_general[i].x : 0;
        faces_general[i].y = faces_general[i].y > 0 ? faces_general[i].y : 0;

        faces_general[i].width += 60;
        faces_general[i].height += 60;
        if (faces_general[i].x + faces_general[i].width >= 640) {
          //printf("x=%d\n", faces_general[i].x + faces_general[i].width) ;
          faces_general[i].width = 640 - faces_general[i].x;

        }
        if (faces_general[i].y + faces_general[i].height >= 480) {
          //printf("y = %d\n", faces_general[i].y + faces_general[i].height) ;
          faces_general[i].height = 480 - faces_general[i].y;
        }
        cv::Mat faceROI = gray_frame(faces_general[i]);


        //-- In each face, detect eyes
        std::vector<cv::Rect> eyes;
        eyes_cascade.detectMultiScale( faceROI, eyes);
        for ( size_t j = 0; j < eyes.size(); j++ ) {
            cv::Point eye_center( faces_general[i].x + eyes[j].x + eyes[j].width/2, faces_general[i].y + eyes[j].y + eyes[j].height/2 );
            int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
            circle( frame, eye_center, radius, cv::Scalar( 255, 0, 0 ), 4 );
        }

        // detect ching's face
        face_cascade_ching.detectMultiScale(faceROI, faces_ching, 0, cv::Size(120,120), cv::Size(480,480), true);
        //printf("faces_ching.size = %d\n", faces_ching.size());

        if (faces_ching.size()) {
          time_t currTime = time(NULL);
          double diff = difftime(currTime, startTime);
          cout <<"Detected Ching Huang, " << diff << " sec" << endl;
          string info = "Ching " + to_string(diff) + " sec";
          putText(frame, info, cv::Point(faces_general[i].x, faces_general[i].y ), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 0, 255));
        }
        else {
          //printf("faces_xuan.size = %d\n", faces_xuan.size());
          face_cascade_xuan.detectMultiScale(gray_frame, faces_xuan, 0, cv::Size(120,120), cv::Size(480,480), true);
          if (faces_xuan.size()) {
            time_t currTime = time(NULL);
            double diff = difftime(currTime, startTime);
            cout <<"Detected YunXuan, " << diff << " sec" << endl;
            string info = "Xuan " to_string(diff) + " sec";
            putText(frame, info, cv::Point(faces_general[i].x, faces_general[i].y ), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 255, 0));
          }
          else {
            std::string name = "Unknown: " + to_string(++unknownID);
            putText(frame, name, cv::Point(faces_general[i].x, faces_general[i].y + faces_general[i].height - 20), FONT_HERSHEY_COMPLEX, 0.5, Scalar(255, 0, 0));

            time_t currTime = time(NULL);
            double diff = difftime(currTime, startTime);
            cout <<"Detected unknown, " << diff << " sec" << endl;
          }
        }

    }
}


void parameterControl(unsigned char key){
    switch(key){
        case 'i':
            shouldDetect = !shouldDetect;
            startTime = time(NULL);
            printf("start detect: %s", ctime(&startTime));
            break;
        default:
            break;
    }

}
