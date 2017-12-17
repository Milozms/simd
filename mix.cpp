#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sys/time.h>
#include <immintrin.h>
#include <stdlib.h>
#include <algorithm>
#include <emmintrin.h>
#include<xmmintrin.h>
using namespace std;

const int WIDTH = 1920;
const int HEIGHT = 1080;


double YuvToArgbMat[3][3] = {1.164383,	0,		1.596027,
                             1.164383,	-0.391762, -0.812968,
                             1.164383,	2.017232,  0};
double ArgbToYuvMat[4][4] = {0.256788,	0.504129,	0.097906,	16,
                             -0.148223,	-0.290993,	0.439216,	128,
                             0.439216,	-0.367788,	-0.071427,	128};
void process_without_simd(char* infile1, char* infile2, char* outfile);
void process_with_mmx(char* infile1, char* infile2, char* outfile);
void process_with_sse(char* infile1, char* infile2, char* outfile);
void process_with_avx(char* infile1, char* infile2, char* outfile);

class Yuv2Argb2Yuv {
public:
    void transmit(unsigned char* y1, unsigned char* u1, unsigned char* v1,
                  unsigned char* y2, unsigned char* u2, unsigned char* v2,
                  unsigned char* yo, unsigned char* uo, unsigned char* vo, int A)
    {
        int temp = 0;
        unsigned char r1, g1, b1, r2, g2, b2, r, g, b;
        for(int y = 0; y < HEIGHT; y++)
        {
            for(int x=0; x < WIDTH; x++)
            {
                //r分量
                temp = (y1[y * WIDTH + x] - 16) * YuvToArgbMat[0][0] + (v1[(y / 2) * (WIDTH / 2) + x / 2] - 128) * YuvToArgbMat[0][2];
                r1 = temp < 0 ? 0 : (temp>255 ? 255 : temp);
                temp = (y2[y * WIDTH + x] - 16) * YuvToArgbMat[0][0] + (v2[(y / 2) * (WIDTH / 2) + x / 2] - 128) * YuvToArgbMat[0][2];
                r2 = temp < 0 ? 0 : (temp>255 ? 255 : temp);
                r = (float)(A * r1 + (256 - A) * r2) / 256;

                //g分量
                temp = (y1[y * WIDTH + x] - 16) * YuvToArgbMat[1][0] + (u1[(y / 2) * (WIDTH / 2) + x / 2] - 128) * YuvToArgbMat[1][1] + (v1[(y / 2) * (WIDTH / 2) + x / 2] - 128) * YuvToArgbMat[1][2];
                g1 = temp < 0 ? 0 : (temp>255 ? 255 : temp);
                temp = (y2[y * WIDTH + x] - 16) * YuvToArgbMat[1][0] + (u2[(y / 2) * (WIDTH / 2) + x / 2] - 128) * YuvToArgbMat[1][1] + (v2[(y / 2) * (WIDTH / 2) + x / 2] - 128) * YuvToArgbMat[1][2];
                g2 = temp < 0 ? 0 : (temp>255 ? 255 : temp);
                g = (float)(A * g1 + (256 - A) * g2) / 256;

                //b分量
                temp = (y1[y * WIDTH + x] - 16) * YuvToArgbMat[2][0] + (u1[( y / 2) * (WIDTH / 2) + x / 2] - 128) * YuvToArgbMat[2][1];
                b1 = temp < 0 ? 0 : (temp>255 ? 255 : temp);
                temp = (y2[y * WIDTH + x] - 16) * YuvToArgbMat[2][0] + (u2[( y / 2) * (WIDTH / 2) + x / 2] - 128) * YuvToArgbMat[2][1];
                b2 = temp < 0 ? 0 : (temp>255 ? 255 : temp);
                b = (float)(A * b1 + (256 - A) * b2) / 256;

                yo[y * WIDTH + x] = r * ArgbToYuvMat[0][0] + g * ArgbToYuvMat[0][1] + b * ArgbToYuvMat[0][2] + ArgbToYuvMat[0][3];
                uo[(y / 2) * (WIDTH / 2) + x / 2] = r * ArgbToYuvMat[1][0] + g * ArgbToYuvMat[1][1] + b * ArgbToYuvMat[1][2] + ArgbToYuvMat[1][3];
                vo[(y / 2) * (WIDTH / 2) + x / 2] = r * ArgbToYuvMat[2][0] + g * ArgbToYuvMat[2][1] + b * ArgbToYuvMat[2][2] + ArgbToYuvMat[2][3];
            }
        }
    }

    void Convert(char *file1, char* file2, char *outputfile, int type) {
        int fReadSize = 0;
        int ImgSize = WIDTH * HEIGHT;
        FILE *fp = NULL;
        unsigned char* cTemp[12];

        int FrameSize = ImgSize + (ImgSize >> 1);
        unsigned char* yuv1 = new unsigned char[FrameSize];
        unsigned char* yuv2 = new unsigned char[FrameSize];
        unsigned char* newyuv = new unsigned char[FrameSize * 85];
        memset(newyuv, 0, FrameSize*85);
        if((fp = fopen(file1, "rb")) == NULL)
            return;
        fReadSize = fread(yuv1, 1, FrameSize, fp);
        fclose(fp);
        if(fReadSize < FrameSize)
            return;

        if((fp = fopen(file2, "rb")) == NULL)
            return;
        fReadSize = fread(yuv2, 1, FrameSize, fp);
        fclose(fp);
        if(fReadSize < FrameSize)
            return;

        cTemp[0] = yuv1;                        //y分量地址
        cTemp[1] = cTemp[0] + ImgSize;            //u分量地址
        cTemp[2] = cTemp[1] + (ImgSize >> 2);    //v分量地址
        cTemp[3] = yuv2;                        //y分量地址
        cTemp[4] = cTemp[3] + ImgSize;            //u分量地址
        cTemp[5] = cTemp[4] + (ImgSize >> 2);    //v分量地址
        cTemp[6] = newyuv;
        cTemp[7] = cTemp[6] + ImgSize;
        cTemp[8] = cTemp[7] + (ImgSize >> 2);


        if((fp = fopen(outputfile, "wb")) == NULL)
            return;
        struct timeval start, end;
        gettimeofday(&start, NULL);
        for (int A = 255; A >= 1; A -= 3)
        {
            switch(type)
            {
                case 0:
                    transmit(cTemp[0], cTemp[1], cTemp[2], cTemp[3], cTemp[4], cTemp[5], cTemp[6], cTemp[7], cTemp[8], A);
                    break;
                case 1:
                    transmitMMX(cTemp[0], cTemp[1], cTemp[2], cTemp[3], cTemp[4], cTemp[5], cTemp[6], cTemp[7], cTemp[8], A);
                    break;
                case 2:
                    transmitSSE2(cTemp[0], cTemp[1], cTemp[2], cTemp[3], cTemp[4], cTemp[5], cTemp[6], cTemp[7], cTemp[8], A);
                    break;
                case 3:
                    transmitAVX(cTemp[0], cTemp[1], cTemp[2], cTemp[3], cTemp[4], cTemp[5], cTemp[6], cTemp[7], cTemp[8], A);
                    break;
                default:
                    cout << "Mode is incorrect, halt.\n" << endl;
                    return;
            }

            cTemp[6] += FrameSize;
            cTemp[7] += FrameSize;
            cTemp[8] += FrameSize;
        }
        gettimeofday(&end, NULL);
        fwrite(newyuv, 1, FrameSize * 85, fp);
        fclose(fp);
        cout << "Time: " << (double)(end.tv_sec-start.tv_sec)*1000.0+(double)(end.tv_usec-start.tv_usec)/1000.0 << "ms.";
        return;
    }
    void transmitMMX(unsigned char* y1, unsigned char* u1, unsigned char* v1,
                     unsigned char* y2, unsigned char* u2, unsigned char* v2,
                     unsigned char* yo, unsigned char* uo, unsigned char* vo, short A)
    {
        __m64 Y2R = _mm_set1_pi16(298);
        __m64 U2R = _mm_set1_pi16(0);
        __m64 V2R = _mm_set1_pi16(409);
        __m64 Y2G = _mm_set1_pi16(298);
        __m64 U2G = _mm_set1_pi16(-100);
        __m64 V2G = _mm_set1_pi16(-208);
        __m64 Y2B = _mm_set1_pi16(298);
        __m64 U2B = _mm_set1_pi16(516);
        __m64 V2B = _mm_set1_pi16(0);
        // >>8

        __m64 R2Y = _mm_set1_pi16(66);
        __m64 G2Y = _mm_set1_pi16(129);
        __m64 B2Y = _mm_set1_pi16(25);
        __m64 R2U = _mm_set1_pi16(-38);
        __m64 G2U = _mm_set1_pi16(-74);
        __m64 B2U = _mm_set1_pi16(112);
        __m64 R2V = _mm_set1_pi16(112);
        __m64 G2V = _mm_set1_pi16(-94);
        __m64 B2V = _mm_set1_pi16(-18);
        //>>8, +mask

        __m64 MAX255 = _mm_set1_pi16(255);
        __m64 MIN0 = _mm_set1_pi16(0);

        __m64 Ymask = _mm_set1_pi16(16);
        __m64 Umask = _mm_set1_pi16(128);
        __m64 Vmask = _mm_set1_pi16(128);

        __m64 mA = _mm_set1_pi16(A);
        __m64 mA2 = _mm_set1_pi16(256 - A);
        for (int y = 0; y < HEIGHT; y++)
        {
            for (int x = 0; x < WIDTH; x += 4)
            {
                __m64 R, G, B, R2, G2, B2;
                __m64 Y, U, V;

                __m64 compareResult, mediaNum;

                __m64 my = _mm_set_pi16(y1[y * WIDTH + x], y1[y * WIDTH + x + 1], y1[y * WIDTH + x + 2], y1[y * WIDTH + x + 3]);
                __m64 mu = _mm_set_pi16(u1[(y / 2) * (WIDTH / 2) + x / 2], u1[(y / 2) * (WIDTH / 2) + (x + 1) / 2],
                                        u1[(y / 2) * (WIDTH / 2) + (x + 2) / 2], u1[(y / 2) * (WIDTH / 2) + (x + 3) / 2]);
                __m64 mv = _mm_set_pi16(v1[(y / 2) * (WIDTH / 2) + x / 2], v1[(y / 2) * (WIDTH / 2) + (x + 1) / 2],
                                        v1[(y / 2) * (WIDTH / 2) + (x + 2) / 2], v1[(y / 2) * (WIDTH / 2) + (x + 3) / 2]);

                // YUV to RGB
                my = _mm_sub_pi16(my, Ymask);
                mu = _mm_sub_pi16(mu, Umask);
                mv = _mm_sub_pi16(mv, Vmask);

                R = _mm_set1_pi16(0);
                G = _mm_set1_pi16(0);
                B = _mm_set1_pi16(0);

                R = _mm_add_pi16(R, _mm_mullo_pi16(Y2R, my));
                R = _mm_add_pi16(R, _mm_mullo_pi16(U2R, mu));
                R = _mm_add_pi16(R, _mm_mullo_pi16(V2R, mv));

                G = _mm_add_pi16(G, _mm_mullo_pi16(Y2G, my));
                G = _mm_add_pi16(G, _mm_mullo_pi16(U2G, mu));
                G = _mm_add_pi16(G, _mm_mullo_pi16(V2G, mv));

                B = _mm_add_pi16(B, _mm_mullo_pi16(Y2B, my));
                B = _mm_add_pi16(B, _mm_mullo_pi16(U2B, mu));
                B = _mm_add_pi16(B, _mm_mullo_pi16(V2B, mv));

                R = _mm_srli_pi16(R, 8);
                G = _mm_srli_pi16(G, 8);
                B = _mm_srli_pi16(B, 8);

                //check range 0~255
                //R>255: 255
                compareResult = _mm_cmpgt_pi16(R, MAX255);
                mediaNum = _mm_sub_pi16(MAX255, R);
                mediaNum = _mm_and_si64(mediaNum, compareResult);
                R = _mm_add_pi16(mediaNum, R);
                //R<0: 0
                compareResult = _mm_cmpgt_pi16(MIN0, R);
                mediaNum = _mm_sub_pi16(MIN0, R);
                mediaNum = _mm_and_si64(mediaNum, compareResult);
                R = _mm_add_pi16(mediaNum, R);
                //G>255: 255
                compareResult = _mm_cmpgt_pi16(G, MAX255);
                mediaNum = _mm_sub_pi16(MAX255, G);
                mediaNum = _mm_and_si64(mediaNum, compareResult);
                G = _mm_add_pi16(mediaNum, G);
                //G<0: 0
                compareResult = _mm_cmpgt_pi16(MIN0, G);
                mediaNum = _mm_sub_pi16(MIN0, G);
                mediaNum = _mm_and_si64(mediaNum, compareResult);
                G = _mm_add_pi16(mediaNum, G);
                //B>255: 255
                compareResult = _mm_cmpgt_pi16(B, MAX255);
                mediaNum = _mm_sub_pi16(MAX255, B);
                mediaNum = _mm_and_si64(mediaNum, compareResult);
                B = _mm_add_pi16(mediaNum, B);
                //B<0: 0
                compareResult = _mm_cmpgt_pi16(MIN0, B);
                mediaNum = _mm_sub_pi16(MIN0, B);
                mediaNum = _mm_and_si64(mediaNum, compareResult);
                B = _mm_add_pi16(mediaNum, B);

                // set alpha
                R = _mm_mullo_pi16(R, mA);
                G = _mm_mullo_pi16(G, mA);
                B = _mm_mullo_pi16(B, mA);
                R = _mm_srli_pi16(R, 8);
                G = _mm_srli_pi16(G, 8);
                B = _mm_srli_pi16(B, 8);

                //image 2
                my = _mm_set_pi16(y2[y * WIDTH + x], y2[y * WIDTH + x + 1], y2[y * WIDTH + x + 2], y2[y * WIDTH + x + 3]);
                mu = _mm_set_pi16(u2[(y / 2) * (WIDTH / 2) + x / 2], u2[(y / 2) * (WIDTH / 2) + (x + 1) / 2],
                                        u2[(y / 2) * (WIDTH / 2) + (x + 2) / 2], u2[(y / 2) * (WIDTH / 2) + (x + 3) / 2]);
                mv = _mm_set_pi16(v2[(y / 2) * (WIDTH / 2) + x / 2], v2[(y / 2) * (WIDTH / 2) + (x + 1) / 2],
                                        v2[(y / 2) * (WIDTH / 2) + (x + 2) / 2], v2[(y / 2) * (WIDTH / 2) + (x + 3) / 2]);

                // YUV to RGB
                my = _mm_sub_pi16(my, Ymask);
                mu = _mm_sub_pi16(mu, Umask);
                mv = _mm_sub_pi16(mv, Vmask);

                R2 = _mm_set1_pi16(0);
                G2 = _mm_set1_pi16(0);
                B2 = _mm_set1_pi16(0);

                R2 = _mm_add_pi16(R2, _mm_mullo_pi16(Y2R, my));
                R2 = _mm_add_pi16(R2, _mm_mullo_pi16(U2R, mu));
                R2 = _mm_add_pi16(R2, _mm_mullo_pi16(V2R, mv));

                G2 = _mm_add_pi16(G2, _mm_mullo_pi16(Y2G, my));
                G2 = _mm_add_pi16(G2, _mm_mullo_pi16(U2G, mu));
                G2 = _mm_add_pi16(G2, _mm_mullo_pi16(V2G, mv));

                B2 = _mm_add_pi16(B2, _mm_mullo_pi16(Y2B, my));
                B2 = _mm_add_pi16(B2, _mm_mullo_pi16(U2B, mu));
                B2 = _mm_add_pi16(B2, _mm_mullo_pi16(V2B, mv));

                R2 = _mm_srli_pi16(R2, 8);
                G2 = _mm_srli_pi16(G2, 8);
                B2 = _mm_srli_pi16(B2, 8);

                //check range 0~255
                //R>255: 255
                compareResult = _mm_cmpgt_pi16(R2, MAX255);
                mediaNum = _mm_sub_pi16(MAX255, R2);
                mediaNum = _mm_and_si64(mediaNum, compareResult);
                R2 = _mm_add_pi16(mediaNum, R2);
                //R<0: 0
                compareResult = _mm_cmpgt_pi16(MIN0, R2);
                mediaNum = _mm_sub_pi16(MIN0, R2);
                mediaNum = _mm_and_si64(mediaNum, compareResult);
                R2 = _mm_add_pi16(mediaNum, R2);
                //G>255: 255
                compareResult = _mm_cmpgt_pi16(G2, MAX255);
                mediaNum = _mm_sub_pi16(MAX255, G2);
                mediaNum = _mm_and_si64(mediaNum, compareResult);
                G2 = _mm_add_pi16(mediaNum, G2);
                //G<0: 0
                compareResult = _mm_cmpgt_pi16(MIN0, G2);
                mediaNum = _mm_sub_pi16(MIN0, G2);
                mediaNum = _mm_and_si64(mediaNum, compareResult);
                G2 = _mm_add_pi16(mediaNum, G2);
                //B>255: 255
                compareResult = _mm_cmpgt_pi16(B2, MAX255);
                mediaNum = _mm_sub_pi16(MAX255, B2);
                mediaNum = _mm_and_si64(mediaNum, compareResult);
                B2 = _mm_add_pi16(mediaNum, B2);
                //B<0: 0
                compareResult = _mm_cmpgt_pi16(MIN0, B2);
                mediaNum = _mm_sub_pi16(MIN0, B2);
                mediaNum = _mm_and_si64(mediaNum, compareResult);
                B2 = _mm_add_pi16(mediaNum, B2);

                // set alpha
                R2 = _mm_mullo_pi16(R2, mA2);
                G2 = _mm_mullo_pi16(G2, mA2);
                B2 = _mm_mullo_pi16(B2, mA2);
                R2 = _mm_srli_pi16(R2, 8);
                G2 = _mm_srli_pi16(G2, 8);
                B2 = _mm_srli_pi16(B2, 8);

                //mix
                R = _mm_add_pi16(R, R2);
                G = _mm_add_pi16(G, G2);
                B = _mm_add_pi16(B, B2);

                // RGB to YUV
                Y = _mm_set1_pi16(0);
                U = _mm_set1_pi16(0);
                V = _mm_set1_pi16(0);

                Y = _mm_add_pi16(Y, _mm_mullo_pi16(R2Y, R));
                Y = _mm_add_pi16(Y, _mm_mullo_pi16(G2Y, G));
                Y = _mm_add_pi16(Y, _mm_mullo_pi16(B2Y, B));

                U = _mm_add_pi16(U, _mm_mullo_pi16(R2U, R));
                U = _mm_add_pi16(U, _mm_mullo_pi16(G2U, G));
                U = _mm_add_pi16(U, _mm_mullo_pi16(B2U, B));

                V = _mm_add_pi16(V, _mm_mullo_pi16(R2V, R));
                V = _mm_add_pi16(V, _mm_mullo_pi16(G2V, G));
                V = _mm_add_pi16(V, _mm_mullo_pi16(B2V, B));

                Y = _mm_srli_pi16(Y, 8);
                U = _mm_srli_pi16(U, 8);
                V = _mm_srli_pi16(V, 8);

                Y = _mm_add_pi16(Y, Ymask);
                U = _mm_add_pi16(U, Umask);
                V = _mm_add_pi16(V, Vmask);

                for (int i = 0; i < 4; i++) {
                    yo[y * WIDTH + x + i] = ((short *)&Y)[3 - i];
                    uo[(y / 2) * (WIDTH / 2) + (x + i) / 2] =  ((short *)&U)[3 - i];
                    vo[(y / 2) * (WIDTH / 2) + (x + i) / 2] =  ((short *)&V)[3 - i];
                }

            }
        }
    }
    void transmitSSE2(unsigned char* y1, unsigned char* u1, unsigned char* v1,
                      unsigned char* y2, unsigned char* u2, unsigned char* v2,
                      unsigned char* yo, unsigned char* uo, unsigned char* vo, short A)
    {
        __m128i Y2R = _mm_set1_epi16(298);
        __m128i U2R = _mm_set1_epi16(0);
        __m128i V2R = _mm_set1_epi16(409);
        __m128i Y2G = _mm_set1_epi16(298);
        __m128i U2G = _mm_set1_epi16(-100);
        __m128i V2G = _mm_set1_epi16(-208);
        __m128i Y2B = _mm_set1_epi16(298);
        __m128i U2B = _mm_set1_epi16(516);
        __m128i V2B = _mm_set1_epi16(0);
        // >>8

        __m128i R2Y = _mm_set1_epi16(66);
        __m128i G2Y = _mm_set1_epi16(129);
        __m128i B2Y = _mm_set1_epi16(25);
        __m128i R2U = _mm_set1_epi16(-38);
        __m128i G2U = _mm_set1_epi16(-74);
        __m128i B2U = _mm_set1_epi16(112);
        __m128i R2V = _mm_set1_epi16(112);
        __m128i G2V = _mm_set1_epi16(-94);
        __m128i B2V = _mm_set1_epi16(-18);
        //>>8, +mask

        __m128i MAX255 = _mm_set1_epi16(255);
        __m128i MIN0 = _mm_set1_epi16(0);

        __m128i Ymask = _mm_set1_epi16(16);
        __m128i Umask = _mm_set1_epi16(128);
        __m128i Vmask = _mm_set1_epi16(128);

        __m128i mA = _mm_set1_epi16(A);
        __m128i mA2 = _mm_set1_epi16(256 - A);
        for (int y = 0; y < HEIGHT; y++)
        {
            for (int x = 0; x < WIDTH; x += 8)
            {
                __m128i R, G, B, R2, G2, B2;
                __m128i Y, U, V;

                __m128i compareResult, mediaNum;

                __m128i my = _mm_set_epi16(y1[y * WIDTH + x], y1[y * WIDTH + x + 1], y1[y * WIDTH + x + 2], y1[y * WIDTH + x + 3],
                                           y1[y * WIDTH + x + 4], y1[y * WIDTH + x + 5], y1[y * WIDTH + x + 6], y1[y * WIDTH + x + 7]);
                __m128i mu = _mm_set_epi16(u1[(y / 2) * (WIDTH / 2) + x / 2], u1[(y / 2) * (WIDTH / 2) + (x + 1) / 2],
                                           u1[(y / 2) * (WIDTH / 2) + (x + 2) / 2], u1[(y / 2) * (WIDTH / 2) + (x + 3) / 2],
                                           u1[(y / 2) * (WIDTH / 2) + (x + 4) / 2], u1[(y / 2) * (WIDTH / 2) + (x + 5) / 2],
                                           u1[(y / 2) * (WIDTH / 2) + (x + 6) / 2], u1[(y / 2) * (WIDTH / 2) + (x + 7) / 2]);
                __m128i mv = _mm_set_epi16(v1[(y / 2) * (WIDTH / 2) + x / 2], v1[(y / 2) * (WIDTH / 2) + (x + 1) / 2],
                                           v1[(y / 2) * (WIDTH / 2) + (x + 2) / 2], v1[(y / 2) * (WIDTH / 2) + (x + 3) / 2],
                                           v1[(y / 2) * (WIDTH / 2) + (x + 4) / 2], v1[(y / 2) * (WIDTH / 2) + (x + 5) / 2],
                                           v1[(y / 2) * (WIDTH / 2) + (x + 6) / 2], v1[(y / 2) * (WIDTH / 2) + (x + 7) / 2]);

                // YUV to RGB
                my = _mm_sub_epi16(my, Ymask);
                mu = _mm_sub_epi16(mu, Umask);
                mv = _mm_sub_epi16(mv, Vmask);

                R = _mm_set1_epi16(0);
                G = _mm_set1_epi16(0);
                B = _mm_set1_epi16(0);

                R = _mm_add_epi16(R, _mm_mullo_epi16(Y2R, my));
                R = _mm_add_epi16(R, _mm_mullo_epi16(U2R, mu));
                R = _mm_add_epi16(R, _mm_mullo_epi16(V2R, mv));

                G = _mm_add_epi16(G, _mm_mullo_epi16(Y2G, my));
                G = _mm_add_epi16(G, _mm_mullo_epi16(U2G, mu));
                G = _mm_add_epi16(G, _mm_mullo_epi16(V2G, mv));

                B = _mm_add_epi16(B, _mm_mullo_epi16(Y2B, my));
                B = _mm_add_epi16(B, _mm_mullo_epi16(U2B, mu));
                B = _mm_add_epi16(B, _mm_mullo_epi16(V2B, mv));

                R = _mm_srli_epi16(R, 8);
                G = _mm_srli_epi16(G, 8);
                B = _mm_srli_epi16(B, 8);

                //check range 0~255
                //R>255: 255
                compareResult = _mm_cmpgt_epi16(R, MAX255);
                mediaNum = _mm_sub_epi16(MAX255, R);
                mediaNum = _mm_and_si128(mediaNum, compareResult);
                R = _mm_add_epi16(mediaNum, R);
                //R<0: 0
                compareResult = _mm_cmplt_epi16(R, MIN0);
                mediaNum = _mm_sub_epi16(MIN0, R);
                mediaNum = _mm_and_si128(mediaNum, compareResult);
                R = _mm_add_epi16(mediaNum, R);
                //G>255: 255
                compareResult = _mm_cmpgt_epi16(G, MAX255);
                mediaNum = _mm_sub_epi16(MAX255, G);
                mediaNum = _mm_and_si128(mediaNum, compareResult);
                G = _mm_add_epi16(mediaNum, G);
                //G<0: 0
                compareResult = _mm_cmplt_epi16(G, MIN0);
                mediaNum = _mm_sub_epi16(MIN0, G);
                mediaNum = _mm_and_si128(mediaNum, compareResult);
                G = _mm_add_epi16(mediaNum, G);
                //B>255: 255
                compareResult = _mm_cmpgt_epi16(B, MAX255);
                mediaNum = _mm_sub_epi16(MAX255, B);
                mediaNum = _mm_and_si128(mediaNum, compareResult);
                B = _mm_add_epi16(mediaNum, B);
                //B<0: 0
                compareResult = _mm_cmplt_epi16(B, MIN0);
                mediaNum = _mm_sub_epi16(MIN0, B);
                mediaNum = _mm_and_si128(mediaNum, compareResult);
                B = _mm_add_epi16(mediaNum, B);

                // set alpha
                R = _mm_mullo_epi16(R, mA);
                G = _mm_mullo_epi16(G, mA);
                B = _mm_mullo_epi16(B, mA);
                R = _mm_srli_epi16(R, 8);
                G = _mm_srli_epi16(G, 8);
                B = _mm_srli_epi16(B, 8);

                //image2
                my = _mm_set_epi16(y2[y * WIDTH + x], y2[y * WIDTH + x + 1], y2[y * WIDTH + x + 2], y2[y * WIDTH + x + 3],
                                           y2[y * WIDTH + x + 4], y2[y * WIDTH + x + 5], y2[y * WIDTH + x + 6], y2[y * WIDTH + x + 7]);
                mu = _mm_set_epi16(u2[(y / 2) * (WIDTH / 2) + x / 2], u2[(y / 2) * (WIDTH / 2) + (x + 1) / 2],
                                           u2[(y / 2) * (WIDTH / 2) + (x + 2) / 2], u2[(y / 2) * (WIDTH / 2) + (x + 3) / 2],
                                           u2[(y / 2) * (WIDTH / 2) + (x + 4) / 2], u2[(y / 2) * (WIDTH / 2) + (x + 5) / 2],
                                           u2[(y / 2) * (WIDTH / 2) + (x + 6) / 2], u2[(y / 2) * (WIDTH / 2) + (x + 7) / 2]);
                mv = _mm_set_epi16(v2[(y / 2) * (WIDTH / 2) + x / 2], v2[(y / 2) * (WIDTH / 2) + (x + 1) / 2],
                                           v2[(y / 2) * (WIDTH / 2) + (x + 2) / 2], v2[(y / 2) * (WIDTH / 2) + (x + 3) / 2],
                                           v2[(y / 2) * (WIDTH / 2) + (x + 4) / 2], v2[(y / 2) * (WIDTH / 2) + (x + 5) / 2],
                                           v2[(y / 2) * (WIDTH / 2) + (x + 6) / 2], v2[(y / 2) * (WIDTH / 2) + (x + 7) / 2]);

                // YUV to RGB
                my = _mm_sub_epi16(my, Ymask);
                mu = _mm_sub_epi16(mu, Umask);
                mv = _mm_sub_epi16(mv, Vmask);

                R2 = _mm_set1_epi16(0);
                G2 = _mm_set1_epi16(0);
                B2 = _mm_set1_epi16(0);

                R2 = _mm_add_epi16(R2, _mm_mullo_epi16(Y2R, my));
                R2 = _mm_add_epi16(R2, _mm_mullo_epi16(U2R, mu));
                R2 = _mm_add_epi16(R2, _mm_mullo_epi16(V2R, mv));

                G2 = _mm_add_epi16(G2, _mm_mullo_epi16(Y2G, my));
                G2 = _mm_add_epi16(G2, _mm_mullo_epi16(U2G, mu));
                G2 = _mm_add_epi16(G2, _mm_mullo_epi16(V2G, mv));

                B2 = _mm_add_epi16(B2, _mm_mullo_epi16(Y2B, my));
                B2 = _mm_add_epi16(B2, _mm_mullo_epi16(U2B, mu));
                B2 = _mm_add_epi16(B2, _mm_mullo_epi16(V2B, mv));

                R2 = _mm_srli_epi16(R2, 8);
                G2 = _mm_srli_epi16(G2, 8);
                B2 = _mm_srli_epi16(B2, 8);

                //check range 0~255
                //R>255: 255
                compareResult = _mm_cmpgt_epi16(R2, MAX255);
                mediaNum = _mm_sub_epi16(MAX255, R2);
                mediaNum = _mm_and_si128(mediaNum, compareResult);
                R2 = _mm_add_epi16(mediaNum, R2);
                //R<0: 0
                compareResult = _mm_cmplt_epi16(R2, MIN0);
                mediaNum = _mm_sub_epi16(MIN0, R2);
                mediaNum = _mm_and_si128(mediaNum, compareResult);
                R2 = _mm_add_epi16(mediaNum, R2);
                //G>255: 255
                compareResult = _mm_cmpgt_epi16(G2, MAX255);
                mediaNum = _mm_sub_epi16(MAX255, G2);
                mediaNum = _mm_and_si128(mediaNum, compareResult);
                G2 = _mm_add_epi16(mediaNum, G2);
                //G<0: 0
                compareResult = _mm_cmplt_epi16(G2, MIN0);
                mediaNum = _mm_sub_epi16(MIN0, G2);
                mediaNum = _mm_and_si128(mediaNum, compareResult);
                G2 = _mm_add_epi16(mediaNum, G2);
                //B>255: 255
                compareResult = _mm_cmpgt_epi16(B2, MAX255);
                mediaNum = _mm_sub_epi16(MAX255, B2);
                mediaNum = _mm_and_si128(mediaNum, compareResult);
                B2 = _mm_add_epi16(mediaNum, B2);
                //B<0: 0
                compareResult = _mm_cmplt_epi16(B2, MIN0);
                mediaNum = _mm_sub_epi16(MIN0, B2);
                mediaNum = _mm_and_si128(mediaNum, compareResult);
                B2 = _mm_add_epi16(mediaNum, B2);

                // set alpha
                R2 = _mm_mullo_epi16(R2, mA2);
                G2 = _mm_mullo_epi16(G2, mA2);
                B2 = _mm_mullo_epi16(B2, mA2);
                R2 = _mm_srli_epi16(R2, 8);
                G2 = _mm_srli_epi16(G2, 8);
                B2 = _mm_srli_epi16(B2, 8);

                //mix
                R = _mm_add_epi16(R, R2);
                G = _mm_add_epi16(G, G2);
                B = _mm_add_epi16(B, B2);

                // RGB to YUV
                Y = _mm_set1_epi16(0);
                U = _mm_set1_epi16(0);
                V = _mm_set1_epi16(0);

                Y = _mm_add_epi16(Y, _mm_mullo_epi16(R2Y, R));
                Y = _mm_add_epi16(Y, _mm_mullo_epi16(G2Y, G));
                Y = _mm_add_epi16(Y, _mm_mullo_epi16(B2Y, B));

                U = _mm_add_epi16(U, _mm_mullo_epi16(R2U, R));
                U = _mm_add_epi16(U, _mm_mullo_epi16(G2U, G));
                U = _mm_add_epi16(U, _mm_mullo_epi16(B2U, B));

                V = _mm_add_epi16(V, _mm_mullo_epi16(R2V, R));
                V = _mm_add_epi16(V, _mm_mullo_epi16(G2V, G));
                V = _mm_add_epi16(V, _mm_mullo_epi16(B2V, B));

                Y = _mm_srli_epi16(Y, 8);
                U = _mm_srli_epi16(U, 8);
                V = _mm_srli_epi16(V, 8);

                Y = _mm_add_epi16(Y, Ymask);
                U = _mm_add_epi16(U, Umask);
                V = _mm_add_epi16(V, Vmask);

                for (int i = 0; i < 8; i++) {
                    yo[y * WIDTH + x + i] = ((short *)&Y)[7 - i];
                    uo[(y / 2) * (WIDTH / 2) + (x + i) / 2] =  ((short *)&U)[7 - i];
                    vo[(y / 2) * (WIDTH / 2) + (x + i) / 2] =  ((short *)&V)[7 - i];
                }

            }
        }
    }
    void transmitAVX(unsigned char* y1, unsigned char* u1, unsigned char* v1,
                     unsigned char* y2, unsigned char* u2, unsigned char* v2,
                     unsigned char* yo, unsigned char* uo, unsigned char* vo, short A)
    {
        __m256i Y2R = _mm256_set1_epi16(298);
        __m256i U2R = _mm256_set1_epi16(0);
        __m256i V2R = _mm256_set1_epi16(409);
        __m256i Y2G = _mm256_set1_epi16(298);
        __m256i U2G = _mm256_set1_epi16(-100);
        __m256i V2G = _mm256_set1_epi16(-208);
        __m256i Y2B = _mm256_set1_epi16(298);
        __m256i U2B = _mm256_set1_epi16(516);
        __m256i V2B = _mm256_set1_epi16(0);
        // >>8

        __m256i R2Y = _mm256_set1_epi16(66);
        __m256i G2Y = _mm256_set1_epi16(129);
        __m256i B2Y = _mm256_set1_epi16(25);
        __m256i R2U = _mm256_set1_epi16(-38);
        __m256i G2U = _mm256_set1_epi16(-74);
        __m256i B2U = _mm256_set1_epi16(112);
        __m256i R2V = _mm256_set1_epi16(112);
        __m256i G2V = _mm256_set1_epi16(-94);
        __m256i B2V = _mm256_set1_epi16(-18);
        //>>8, +mask

        __m256i MAX255 = _mm256_set1_epi16(255);
        __m256i MIN0 = _mm256_set1_epi16(0);

        __m256i Ymask = _mm256_set1_epi16(16);
        __m256i Umask = _mm256_set1_epi16(128);
        __m256i Vmask = _mm256_set1_epi16(128);

        __m256i mA = _mm256_set1_epi16(A);
        __m256i mA2 = _mm256_set1_epi16(256 - A);
        for (int y = 0; y < HEIGHT; y++)
        {
            for (int x = 0; x < WIDTH; x += 16)
            {
                __m256i R, G, B, R2, G2, B2;
                __m256i Y, U, V;

                __m256i compareResult, mediaNum;

                __m256i my = _mm256_set_epi16(y1[y * WIDTH + x], y1[y * WIDTH + x + 1], y1[y * WIDTH + x + 2], y1[y * WIDTH + x + 3],
                                              y1[y * WIDTH + x + 4], y1[y * WIDTH + x + 5], y1[y * WIDTH + x + 6], y1[y * WIDTH + x + 7],
                                              y1[y * WIDTH + x + 8], y1[y * WIDTH + x + 9], y1[y * WIDTH + x + 10], y1[y * WIDTH + x + 11],
                                              y1[y * WIDTH + x + 12], y1[y * WIDTH + x + 13], y1[y * WIDTH + x + 14], y1[y * WIDTH + x + 15]);
                __m256i mu = _mm256_set_epi16(u1[(y / 2) * (WIDTH / 2) + x / 2], u1[(y / 2) * (WIDTH / 2) + (x + 1) / 2],
                                              u1[(y / 2) * (WIDTH / 2) + (x + 2) / 2], u1[(y / 2) * (WIDTH / 2) + (x + 3) / 2],
                                              u1[(y / 2) * (WIDTH / 2) + (x + 4) / 2], u1[(y / 2) * (WIDTH / 2) + (x + 5) / 2],
                                              u1[(y / 2) * (WIDTH / 2) + (x + 6) / 2], u1[(y / 2) * (WIDTH / 2) + (x + 7) / 2],
                                              u1[(y / 2) * (WIDTH / 2) + (x + 8) / 2], u1[(y / 2) * (WIDTH / 2) + (x + 9) / 2],
                                              u1[(y / 2) * (WIDTH / 2) + (x + 10) / 2], u1[(y / 2) * (WIDTH / 2) + (x + 11) / 2],
                                              u1[(y / 2) * (WIDTH / 2) + (x + 12) / 2], u1[(y / 2) * (WIDTH / 2) + (x + 13) / 2],
                                              u1[(y / 2) * (WIDTH / 2) + (x + 14) / 2], u1[(y / 2) * (WIDTH / 2) + (x + 15) / 2]);
                __m256i mv = _mm256_set_epi16(v1[(y / 2) * (WIDTH / 2) + x / 2], v1[(y / 2) * (WIDTH / 2) + (x + 1) / 2],
                                              v1[(y / 2) * (WIDTH / 2) + (x + 2) / 2], v1[(y / 2) * (WIDTH / 2) + (x + 3) / 2],
                                              v1[(y / 2) * (WIDTH / 2) + (x + 4) / 2], v1[(y / 2) * (WIDTH / 2) + (x + 5) / 2],
                                              v1[(y / 2) * (WIDTH / 2) + (x + 6) / 2], v1[(y / 2) * (WIDTH / 2) + (x + 7) / 2],
                                              v1[(y / 2) * (WIDTH / 2) + (x + 8) / 2], v1[(y / 2) * (WIDTH / 2) + (x + 9) / 2],
                                              v1[(y / 2) * (WIDTH / 2) + (x + 10) / 2], v1[(y / 2) * (WIDTH / 2) + (x + 11) / 2],
                                              v1[(y / 2) * (WIDTH / 2) + (x + 12) / 2], v1[(y / 2) * (WIDTH / 2) + (x + 13) / 2],
                                              v1[(y / 2) * (WIDTH / 2) + (x + 14) / 2], v1[(y / 2) * (WIDTH / 2) + (x + 15) / 2]);

                // YUV to RGB
                my = _mm256_sub_epi16(my, Ymask);
                mu = _mm256_sub_epi16(mu, Umask);
                mv = _mm256_sub_epi16(mv, Vmask);

                R = _mm256_set1_epi16(0);
                G = _mm256_set1_epi16(0);
                B = _mm256_set1_epi16(0);

                R = _mm256_add_epi16(R, _mm256_mullo_epi16(Y2R, my));
                R = _mm256_add_epi16(R, _mm256_mullo_epi16(U2R, mu));
                R = _mm256_add_epi16(R, _mm256_mullo_epi16(V2R, mv));

                G = _mm256_add_epi16(G, _mm256_mullo_epi16(Y2G, my));
                G = _mm256_add_epi16(G, _mm256_mullo_epi16(U2G, mu));
                G = _mm256_add_epi16(G, _mm256_mullo_epi16(V2G, mv));

                B = _mm256_add_epi16(B, _mm256_mullo_epi16(Y2B, my));
                B = _mm256_add_epi16(B, _mm256_mullo_epi16(U2B, mu));
                B = _mm256_add_epi16(B, _mm256_mullo_epi16(V2B, mv));

                R = _mm256_srli_epi16(R, 8);
                G = _mm256_srli_epi16(G, 8);
                B = _mm256_srli_epi16(B, 8);

                //check range 0~255
                //R>255: 255
                compareResult = _mm256_cmpgt_epi16(R, MAX255);
                mediaNum = _mm256_sub_epi16(MAX255, R);
                mediaNum = _mm256_and_si256(mediaNum, compareResult);
                R = _mm256_add_epi16(mediaNum, R);
                //R<0: 0
                compareResult = _mm256_cmpgt_epi16(MIN0, R);
                mediaNum = _mm256_sub_epi16(MIN0, R);
                mediaNum = _mm256_and_si256(mediaNum, compareResult);
                R = _mm256_add_epi16(mediaNum, R);
                //G>255: 255
                compareResult = _mm256_cmpgt_epi16(G, MAX255);
                mediaNum = _mm256_sub_epi16(MAX255, G);
                mediaNum = _mm256_and_si256(mediaNum, compareResult);
                G = _mm256_add_epi16(mediaNum, G);
                //G<0: 0
                compareResult = _mm256_cmpgt_epi16(MIN0, G);
                mediaNum = _mm256_sub_epi16(MIN0, G);
                mediaNum = _mm256_and_si256(mediaNum, compareResult);
                G = _mm256_add_epi16(mediaNum, G);
                //B>255: 255
                compareResult = _mm256_cmpgt_epi16(B, MAX255);
                mediaNum = _mm256_sub_epi16(MAX255, B);
                mediaNum = _mm256_and_si256(mediaNum, compareResult);
                B = _mm256_add_epi16(mediaNum, B);
                //B<0: 0
                compareResult = _mm256_cmpgt_epi16(MIN0, B);
                mediaNum = _mm256_sub_epi16(MIN0, B);
                mediaNum = _mm256_and_si256(mediaNum, compareResult);
                B = _mm256_add_epi16(mediaNum, B);

                // set alpha
                R = _mm256_mullo_epi16(R, mA);
                G = _mm256_mullo_epi16(G, mA);
                B = _mm256_mullo_epi16(B, mA);
                R = _mm256_srli_epi16(R, 8);
                G = _mm256_srli_epi16(G, 8);
                B = _mm256_srli_epi16(B, 8);

                //image2
                my = _mm256_set_epi16(y2[y * WIDTH + x], y2[y * WIDTH + x + 1], y2[y * WIDTH + x + 2], y2[y * WIDTH + x + 3],
                                      y2[y * WIDTH + x + 4], y2[y * WIDTH + x + 5], y2[y * WIDTH + x + 6], y2[y * WIDTH + x + 7],
                                      y2[y * WIDTH + x + 8], y2[y * WIDTH + x + 9], y2[y * WIDTH + x + 10], y2[y * WIDTH + x + 11],
                                      y2[y * WIDTH + x + 12], y2[y * WIDTH + x + 13], y2[y * WIDTH + x + 14], y2[y * WIDTH + x + 15]);
                mu = _mm256_set_epi16(u2[(y / 2) * (WIDTH / 2) + x / 2], u2[(y / 2) * (WIDTH / 2) + (x + 1) / 2],
                                              u2[(y / 2) * (WIDTH / 2) + (x + 2) / 2], u2[(y / 2) * (WIDTH / 2) + (x + 3) / 2],
                                              u2[(y / 2) * (WIDTH / 2) + (x + 4) / 2], u2[(y / 2) * (WIDTH / 2) + (x + 5) / 2],
                                              u2[(y / 2) * (WIDTH / 2) + (x + 6) / 2], u2[(y / 2) * (WIDTH / 2) + (x + 7) / 2],
                                              u2[(y / 2) * (WIDTH / 2) + (x + 8) / 2], u2[(y / 2) * (WIDTH / 2) + (x + 9) / 2],
                                              u2[(y / 2) * (WIDTH / 2) + (x + 10) / 2], u2[(y / 2) * (WIDTH / 2) + (x + 11) / 2],
                                              u2[(y / 2) * (WIDTH / 2) + (x + 12) / 2], u2[(y / 2) * (WIDTH / 2) + (x + 13) / 2],
                                              u2[(y / 2) * (WIDTH / 2) + (x + 14) / 2], u2[(y / 2) * (WIDTH / 2) + (x + 15) / 2]);
                mv = _mm256_set_epi16(v2[(y / 2) * (WIDTH / 2) + x / 2], v2[(y / 2) * (WIDTH / 2) + (x + 1) / 2],
                                              v2[(y / 2) * (WIDTH / 2) + (x + 2) / 2], v2[(y / 2) * (WIDTH / 2) + (x + 3) / 2],
                                              v2[(y / 2) * (WIDTH / 2) + (x + 4) / 2], v2[(y / 2) * (WIDTH / 2) + (x + 5) / 2],
                                              v2[(y / 2) * (WIDTH / 2) + (x + 6) / 2], v2[(y / 2) * (WIDTH / 2) + (x + 7) / 2],
                                              v2[(y / 2) * (WIDTH / 2) + (x + 8) / 2], v2[(y / 2) * (WIDTH / 2) + (x + 9) / 2],
                                              v2[(y / 2) * (WIDTH / 2) + (x + 10) / 2], v2[(y / 2) * (WIDTH / 2) + (x + 11) / 2],
                                              v2[(y / 2) * (WIDTH / 2) + (x + 12) / 2], v2[(y / 2) * (WIDTH / 2) + (x + 13) / 2],
                                              v2[(y / 2) * (WIDTH / 2) + (x + 14) / 2], v2[(y / 2) * (WIDTH / 2) + (x + 15) / 2]);

                // YUV to RGB
                my = _mm256_sub_epi16(my, Ymask);
                mu = _mm256_sub_epi16(mu, Umask);
                mv = _mm256_sub_epi16(mv, Vmask);

                R2 = _mm256_set1_epi16(0);
                G2 = _mm256_set1_epi16(0);
                B2 = _mm256_set1_epi16(0);

                R2 = _mm256_add_epi16(R2, _mm256_mullo_epi16(Y2R, my));
                R2 = _mm256_add_epi16(R2, _mm256_mullo_epi16(U2R, mu));
                R2 = _mm256_add_epi16(R2, _mm256_mullo_epi16(V2R, mv));

                G2 = _mm256_add_epi16(G2, _mm256_mullo_epi16(Y2G, my));
                G2 = _mm256_add_epi16(G2, _mm256_mullo_epi16(U2G, mu));
                G2 = _mm256_add_epi16(G2, _mm256_mullo_epi16(V2G, mv));

                B2 = _mm256_add_epi16(B2, _mm256_mullo_epi16(Y2B, my));
                B2 = _mm256_add_epi16(B2, _mm256_mullo_epi16(U2B, mu));
                B2 = _mm256_add_epi16(B2, _mm256_mullo_epi16(V2B, mv));

                R2 = _mm256_srli_epi16(R2, 8);
                G2 = _mm256_srli_epi16(G2, 8);
                B2 = _mm256_srli_epi16(B2, 8);

                //check range 0~255
                //R>255: 255
                compareResult = _mm256_cmpgt_epi16(R2, MAX255);
                mediaNum = _mm256_sub_epi16(MAX255, R2);
                mediaNum = _mm256_and_si256(mediaNum, compareResult);
                R2 = _mm256_add_epi16(mediaNum, R2);
                //R<0: 0
                compareResult = _mm256_cmpgt_epi16(MIN0, R2);
                mediaNum = _mm256_sub_epi16(MIN0, R2);
                mediaNum = _mm256_and_si256(mediaNum, compareResult);
                R2 = _mm256_add_epi16(mediaNum, R2);
                //G>255: 255
                compareResult = _mm256_cmpgt_epi16(G2, MAX255);
                mediaNum = _mm256_sub_epi16(MAX255, G2);
                mediaNum = _mm256_and_si256(mediaNum, compareResult);
                G2 = _mm256_add_epi16(mediaNum, G2);
                //G<0: 0
                compareResult = _mm256_cmpgt_epi16(MIN0, G2);
                mediaNum = _mm256_sub_epi16(MIN0, G2);
                mediaNum = _mm256_and_si256(mediaNum, compareResult);
                G2 = _mm256_add_epi16(mediaNum, G2);
                //B>255: 255
                compareResult = _mm256_cmpgt_epi16(B2, MAX255);
                mediaNum = _mm256_sub_epi16(MAX255, B2);
                mediaNum = _mm256_and_si256(mediaNum, compareResult);
                B2 = _mm256_add_epi16(mediaNum, B2);
                //B<0: 0
                compareResult = _mm256_cmpgt_epi16(MIN0, B2);
                mediaNum = _mm256_sub_epi16(MIN0, B2);
                mediaNum = _mm256_and_si256(mediaNum, compareResult);
                B2 = _mm256_add_epi16(mediaNum, B2);

                // set alpha
                R2 = _mm256_mullo_epi16(R2, mA2);
                G2 = _mm256_mullo_epi16(G2, mA2);
                B2 = _mm256_mullo_epi16(B2, mA2);
                R2 = _mm256_srli_epi16(R2, 8);
                G2 = _mm256_srli_epi16(G2, 8);
                B2 = _mm256_srli_epi16(B2, 8);

                //mix
                R = _mm256_add_epi16(R, R2);
                G = _mm256_add_epi16(G, G2);
                B = _mm256_add_epi16(B, B2);

                // RGB to YUV
                Y = _mm256_set1_epi16(0);
                U = _mm256_set1_epi16(0);
                V = _mm256_set1_epi16(0);

                Y = _mm256_add_epi16(Y, _mm256_mullo_epi16(R2Y, R));
                Y = _mm256_add_epi16(Y, _mm256_mullo_epi16(G2Y, G));
                Y = _mm256_add_epi16(Y, _mm256_mullo_epi16(B2Y, B));

                U = _mm256_add_epi16(U, _mm256_mullo_epi16(R2U, R));
                U = _mm256_add_epi16(U, _mm256_mullo_epi16(G2U, G));
                U = _mm256_add_epi16(U, _mm256_mullo_epi16(B2U, B));

                V = _mm256_add_epi16(V, _mm256_mullo_epi16(R2V, R));
                V = _mm256_add_epi16(V, _mm256_mullo_epi16(G2V, G));
                V = _mm256_add_epi16(V, _mm256_mullo_epi16(B2V, B));

                Y = _mm256_srli_epi16(Y, 8);
                U = _mm256_srli_epi16(U, 8);
                V = _mm256_srli_epi16(V, 8);

                Y = _mm256_add_epi16(Y, Ymask);
                U = _mm256_add_epi16(U, Umask);
                V = _mm256_add_epi16(V, Vmask);

                for (int i = 0; i < 16; i++) {
                    yo[y * WIDTH + x + i] = ((short *)&Y)[15 - i];
                    uo[(y / 2) * (WIDTH / 2) + (x + i) / 2] =  ((short *)&U)[15 - i];
                    vo[(y / 2) * (WIDTH / 2) + (x + i) / 2] =  ((short *)&V)[15 - i];
                }

            }
        }
    }
};



void process_without_simd(char* infile1, char* infile2, char* outfile){
    cout<< "Processing without SIMD."<<endl;
    Yuv2Argb2Yuv example;
    example.Convert(infile1, infile2,  outfile,  0);
    cout << endl <<outfile<< " is created" << endl;

}

void process_with_mmx(char* infile1, char* infile2, char* outfile){
    cout<< "Processing with MMX."<<endl;
    Yuv2Argb2Yuv example;
    example.Convert(infile1, infile2, outfile,  1);
    cout << endl <<outfile<< " is created" << endl;
}

void process_with_sse(char* infile1, char* infile2, char* outfile){
    cout<< "Processing with SSE2."<<endl;
    Yuv2Argb2Yuv example;
    example.Convert(infile1, infile2, outfile,  2);
    cout << endl <<outfile<< " is created" << endl;
}

void process_with_avx(char* infile1, char* infile2, char* outfile){
    cout<< "Processing with AVX."<<endl;
    Yuv2Argb2Yuv example;
    example.Convert(infile1, infile2, outfile,  3);
    cout << endl <<outfile<< " is created" << endl;
}


int main(int argc, char* argv[]){
    if(argc > 4){
        char *infile1 = argv[1], *infile2 = argv[2], *outfile = argv[2];
        switch(argv[3][0]){
            case 'm':
                process_with_mmx(infile1, infile2, outfile);
                break;
            case 's':
                process_with_sse(infile1, infile2, outfile);
                break;
            case 'a':
                process_with_avx(infile1, infile2, outfile);
                break;
            default:
                cout<<"Argument Error."<<endl;
        }
    }
    else {
        process_without_simd("../image/dem1.yuv", "../image/dem2.yuv", "../image/MIX_ref.yuv");
        process_with_mmx("../image/dem1.yuv", "../image/dem2.yuv", "../image/MIX_MMX.yuv");
        process_with_sse("../image/dem1.yuv", "../image/dem2.yuv", "../image/MIX_SSE.yuv");
        process_with_avx("../image/dem1.yuv", "../image/dem2.yuv", "../image/MIX_AVX.yuv");
    }
    return 0;
}