#include <opencv2/opencv.hpp>
#include <fcntl.h>
#include <cstring>
#include "Retrans.hpp"
#include "mode.hpp"

#ifdef _WIN32//win32下有图形界面，自动退化到imshow

void imshow_self(const cv::Mat& frame)
{
#ifdef WITH_IMSHOW
    return;
#else
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(640, 480), 0, 0, cv::INTER_AREA);
    cv::imshow("output", resized);
    cv::waitKey(1);
#endif
}

#else//linux下直接劫持转发

#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>

class VirtualCamera {
private:
    int v4l2_fd;
    int width, height;

public:
    VirtualCamera(const char* device_path, int w, int h) : width(w), height(h) {
        v4l2_fd = open(device_path, O_RDWR);
        if (v4l2_fd < 0) return;

        struct v4l2_format vfmt;
        memset(&vfmt, 0, sizeof(vfmt));
        vfmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
        vfmt.fmt.pix.width = width;
        vfmt.fmt.pix.height = height;
        vfmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
        vfmt.fmt.pix.sizeimage = width * height * 2;
        vfmt.fmt.pix.field = V4L2_FIELD_NONE;

        if (ioctl(v4l2_fd, VIDIOC_S_FMT, &vfmt) < 0) {
            close(v4l2_fd);
            v4l2_fd = -1;
        }
    }

    ~VirtualCamera() {
        if (v4l2_fd >= 0) close(v4l2_fd);
    }

    bool writeFrame(const cv::Mat& yuyv_frame) {
        if (v4l2_fd < 0) return false;
        ssize_t written = write(v4l2_fd, yuyv_frame.data, yuyv_frame.total() * yuyv_frame.elemSize());
        return written > 0;
    }

    bool isValid() const { return v4l2_fd >= 0; }
};

// BGR转YUYV
void bgrToYUYV(const cv::Mat& bgr, cv::Mat& yuyv) {
    int width = bgr.cols;
    int height = bgr.rows;
    yuyv.create(height, width, CV_8UC2);

    for (int y = 0; y < height; ++y) {
        const uchar* bgr_ptr = bgr.ptr<uchar>(y);
        uchar* yuyv_ptr = yuyv.ptr<uchar>(y);

        for (int x = 0; x < width; x += 2) {
            uchar b1 = bgr_ptr[3 * x], g1 = bgr_ptr[3 * x + 1], r1 = bgr_ptr[3 * x + 2];
            uchar b2 = bgr_ptr[3 * (x + 1)], g2 = bgr_ptr[3 * (x + 1) + 1], r2 = bgr_ptr[3 * (x + 1) + 2];

            uchar y1 = static_cast<uchar>(0.299 * r1 + 0.587 * g1 + 0.114 * b1);
            uchar u1 = static_cast<uchar>(-0.147 * r1 - 0.289 * g1 + 0.436 * b1 + 128);
            uchar y2 = static_cast<uchar>(0.299 * r2 + 0.587 * g2 + 0.114 * b2);
            uchar v1 = static_cast<uchar>(0.615 * r1 - 0.515 * g1 - 0.100 * b1 + 128);

            yuyv_ptr[2 * x] = y1;
            yuyv_ptr[2 * x + 1] = u1;
            yuyv_ptr[2 * x + 2] = y2;
            yuyv_ptr[2 * x + 3] = v1;
        }
    }
}

// 全局虚拟摄像头实例
static VirtualCamera vcam("/dev/video20", 640, 480);

void imshow_self(const cv::Mat& frame) {
    static cv::Mat resized_frame, yuyv_frame;

    // 调整尺寸到640x480
    cv::resize(frame, resized_frame, cv::Size(640, 480));

    // 转换为YUYV格式并写入虚拟摄像头
    bgrToYUYV(resized_frame, yuyv_frame);
    vcam.writeFrame(yuyv_frame);
}
#endif