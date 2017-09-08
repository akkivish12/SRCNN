/***
 * Image Super-Resolution using Deep Convolutional Networks 
 *  @author Ameet
 ***/

#include "stdafx.h"

int main(int argc, char** argv) {
	/* Read the original image */
	Mat img_original;
	//img_original = imread("input/butterfly_GT.bmp");
	string inputfilename = argv[1];
	string outputfilename = inputfilename.substr(0, inputfilename.find_last_of("."));
	outputfilename.append("_output.jpg");
	img_original = imread(inputfilename);
	cout << "Read the Original Image Successfully ..." << endl;
	
	/* convert RGB img to YCrCb */
	Mat img_YCrCb;
	cout << "Convert the Image to YCrCb Sucessfully ..." << endl;
	cvtColor(img_original, img_YCrCb, CV_BGR2YCrCb);
	vector<Mat> img_YCrCbCh(3);
	cout << "Spliting the Y-Cr-Cb Channel ..." << endl;
	split(img_YCrCb, img_YCrCbCh);
	
	vector<Mat> img_vec(3);
	for (int i = 0; i < 3; i++) {
		resize(img_YCrCbCh[i], img_vec[i], img_YCrCbCh[i].size()*UP_SCALE, 0, 0, CV_INTER_CUBIC);
	}
	cout << "Completed Bicubic Interpolation ..." << endl;

	Mat img_cubic;
	resize(img_YCrCb, img_cubic, img_YCrCb.size()*UP_SCALE, 0, 0, CV_INTER_CUBIC);
	cvtColor(img_cubic, img_cubic, CV_YCrCb2BGR);
	//imshow("Cubic", img_cubic);
	imwrite("output/bicubic.bmp", img_cubic);

	/* convolution of Layer1 */
	vector<Mat> img_conv1(CONV1_FILTERS);
	for (int i = 0; i < CONV1_FILTERS; i++) {
		img_conv1[i].create(img_vec[0].size(), CV_32F);
		conv_f1(img_vec[0], img_conv1[i], weights_conv1_data[i], biases_conv1[i]);
	}
	//imshow("Conv1", img_conv1[8]);
	//imwrite("output/Conv1.bmp", img_conv1[8]);
	
	/* convolution of Layer2 */
	vector<Mat> img_conv2(CONV2_FILTERS);
	for (int i = 0; i < CONV2_FILTERS; i++) {
		img_conv2[i].create(img_vec[0].size(), CV_32F);
		conv_f2(img_conv1, img_conv2[i], weights_conv2_data[i], biases_conv2[i]);
	}
	//imshow("Conv2", img_conv2[31]);
	//imwrite("output/Conv2.bmp", img_conv2[31]);
	
	/* convolution of Layer3 */	
	Mat img_conv3;
	img_conv3.create(img_vec[0].size(), CV_8U);
	conv_f3(img_conv2, img_conv3, weights_conv3_data, biases_conv3);
	//imshow("Conv3", img_conv3);
	//imwrite("output/Conv3.bmp", img_conv3);
	
	/* Merge the Y-Cr-Cb Channel back into an image */
	Mat img_YCrCb_out;
	merge(img_vec, img_YCrCb_out);
	cout << "Merge Image Complete..." << endl;

	/* Convert the image from YCrCb to BGR Space */
	Mat img_BGR_out;
	cvtColor(img_YCrCb_out, img_BGR_out, CV_YCrCb2BGR);
	//imshow("Output", img_BGR_out);
	imwrite(outputfilename, img_BGR_out);
	cout << "Convert the Image to BGR Sucessfully ..." << endl;
	//waitKey(0);
	//destroyAllWindows();
	return 0;
}


void conv_f1(Mat& src, Mat& dst, float kernel[9][9], float bias) {
	/* padding of the src image */
	Mat src2;
	src2.create(Size(src.cols + 8, src.rows + 8), CV_8U);
	
	for (int row = 0; row < src2.rows; row++) {
		for (int col = 0; col < src2.cols; col++) {
			int tmpRow = row - 4;
			int tmpCol = col - 4;
			
			if (tmpRow < 0)
				tmpRow = 0;
			else if (tmpRow >= src.rows)
				tmpRow = src.rows - 1;

			if (tmpCol < 0)
				tmpCol = 0;
			else if (tmpCol >= src.cols)
				tmpCol = src.cols - 1;

			src2.at<uchar>(row, col) = src.at<uchar>(tmpRow, tmpCol);
		}
	}
	//imshow("Src2", src2);
	/* Convolution 1 */
	for (int row = 0; row < dst.rows; row++) {
		for (int col = 0; col < dst.cols; col++) {
			float temp = 0;
			for (int i = 0; i < 9; i++) {
				for (int j = 0; j < 9; j++) {
					temp += kernel[i][j] * src2.at<uchar>(row + i, col + j);
				}
			}
			temp += bias;
			temp = (temp >= 0) ? temp : 0; /* Threshold */
			dst.at<float>(row, col) = temp;
		}
	}
	return;
}

void conv_f2(vector<Mat>& src, Mat& dst, float kernel[CONV1_FILTERS], float bias) {
	for (int row = 0; row < dst.rows; row++) {
		for (int col = 0; col < dst.cols; col++) {
			float temp = 0;
			for (int i = 0; i < CONV1_FILTERS; i++) {
				temp += src[i].at<float>(row, col) * kernel[i];
			}
			temp += bias;
			temp = (temp >= 0) ? temp : 0; /* Threshold */
			dst.at<float>(row, col) = temp;
		}
	}
	return;
}

void conv_f3(vector<Mat>& src, Mat& dst, float kernel[32][5][5], float bias) {
	/* padding of the src image */
	vector<Mat> src2(CONV2_FILTERS);
	for (int i = 0; i < CONV2_FILTERS; i++) {
		src2[i].create(Size(src[i].cols + 4, src[i].rows + 4), CV_32F);
		for (int row = 0; row < src2[i].rows; row++) {
			for (int col = 0; col < src2[i].cols; col++) {
				int tmpRow = row - 2;
				int tmpCol = col - 2;

				if (tmpRow < 0)
					tmpRow = 0;
				else if (tmpRow >= src[i].rows)
					tmpRow = src[i].rows - 1;

				if (tmpCol < 0)
					tmpCol = 0;
				else if (tmpCol >= src[i].cols)
					tmpCol = src[i].cols - 1;

				src2[i].at<float>(row, col) = src[i].at<float>(tmpRow, tmpCol);
			}
		}
	}

	/* Convolution 2 */
	for (int row = 0; row < dst.rows; row++) {
		for (int col = 0; col < dst.cols; col++) {
			float temp = 0;
			for (int i = 0; i < CONV2_FILTERS; i++) {
				double temppixel = 0;
				for (int m = 0; m < 5; m++) {
					for (int n = 0; n < 5; n++) {
						temppixel += kernel[i][m][n] * src2[i].at<float>(row + m, col + n);
					}
				}
				temp += temppixel;
			}
			temp += bias;
			temp = (temp >= 0) ? temp : 0;  /* Threshold */
			temp = (temp <= 255) ? temp : 255; /* Threshold */
			dst.at<uchar>(row, col) = (uchar)temp;
		}
	}
	return;
}
