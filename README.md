<div align="right">
  <a href="./README_zh.md" style="text-decoration:none">
    <img src="https://img.shields.io/badge/中文-切换语言-blue?style=for-the-badge" alt="切换到中文">
  </a>
</div>

# Faster-TrackFieldLaneDetect

This project provides solutions for track line detection in competition scenarios.

## Functions

This project captures gradient information on the track using a self-developed NMS-free multi-directional gradient Canny operator. Combined with Hough Transform and DBSCAN clustering, it achieves robust and fast lane line recognition and tracking. It is partially immune to strong light reflections and uneven lighting conditions. 

Extensive use of double buffering mechanisms with lock-free read/write operations optimizes performance. The bare project runs at 60fps-70fps on a Raspberry Pi 4B, making it suitable for related competitions. It supports cross-platform use directly in Windows/Linux environments. When running on Windows, it reads the sample video from the `img` folder; on Linux, it accesses camera 0 for real-time processing.

## Warnings

This project makes extensive use of thread schedulers. Please ensure your processor has at least 4 cores, or manually modify the code to reduce the number of threads. Otherwise, performance degradation or even failure may occur.

This project uses the C++20 standard. If your compiler supports only C++17 or C++14 standards, please replace the coroutine implementation with the Boost library implementation or directly disable the corresponding code. This part is primarily used for state machine scheduling and does not affect the vision processing functionality.

## How to Use

- **Using Functional Modules**: If you wish to use the functional modules in this project, the individual files all support Linux/Windows compilation. You can directly copy the corresponding `.cpp` and `.hpp` files.
- **Running the Complete Project**: If you wish to run the full project directly, please install VS2022 for building the project first.
  1. Right-click to open a terminal in the directory where you want to store the project, and enter the following command to clone the code repository:
     ```bash
     git clone https://github.com/PPlayer404/Faster-TrackFieldLaneDetect.git
     ```
  2. Then open the project's `.sln` file, select Release mode, and configure OpenCV version 4.1.0+ for the Release mode yourself. After that, select build and run.
- **Configuration Options**: In `mode.hpp`, you can choose:
  - Whether to enable `imshow` display (enabled by default)
  - Whether to enable Kalman filter result display (disabled by default)
  - Whether to enable slow debugging (enabled by default)

## License

This project is open-sourced under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html) license.

For any questions, please submit an Issue or contact the author via email: 439887968@qq.com
