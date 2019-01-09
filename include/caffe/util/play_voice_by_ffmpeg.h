#ifndef PLAY_VOICE_BY_FFMPEG_H
#define PLAY_VOICE_BY_FFMPEG_H
/**
 * 最简单的基于FFmpeg的音频播放器  1.2
 * Simplest FFmpeg Audio Player  1.2
 *
 * 雷霄骅 Lei Xiaohua
 * leixiaohua1020@126.com
 * 中国传媒大学/数字电视技术
 * Communication University of China / Digital TV Technology
 * http://blog.csdn.net/leixiaohua1020
 *
 * 本程序实现了音频的解码和播放。
 *
 * This software decode and play audio streams.
 */


extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include <libavutil/mathematics.h>
#include <libavutil/samplefmt.h>
#include<libswresample/swresample.h>

//SDL
}

#include <SDL.h>
#include <SDL_thread.h>
#include <stdlib.h>
#include <string.h>
#include<iostream>
using namespace std;




#define MAX_AUDIO_FRAME_SIZE 192000 // 1 second of 48khz 32bit audio

//Output PCM
#define OUTPUT_PCM 1
//Use SDL
#define USE_SDL 1

//Buffer:
//|-----------|-------------|
//chunk-------pos---len-----|
static  Uint8  *audio_chunk;
static  Uint32  audio_len;
static  Uint8  *audio_pos;



namespace ffmpegs
{

//播放语音的主程序
int play_voice(string voicepath);
}
#endif // PLAY_VOICE_BY_FFMPEG_H
