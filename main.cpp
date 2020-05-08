#include "mbed.h"
#include <cmath>
#include <string>
#include <iostream>
#include "DA7212.h"
#include "uLCD_4DGL.h"
#define NUM_OF_SONGS 2
enum Mode {
    PLAY, MODE_SELECTION, SONG_SELECTION
};
DA7212 audio;
Serial pc(USBTX, USBRX);
int16_t waveform[kAudioTxBufferSize];
EventQueue note_queue(32 * EsVENTS_EVENT_SIZE);
uLCD_4DGL uLCD(D1, D0, D2);
InterruptIn button(SW2);
int id;
int flag = 1;
int song_iter = 0;
int mode_iter = 0;
bool is_changed = false;
Mode mode;
Thread note_thread;
int song[NUM_OF_SONGS][42] = {
    {261, 261, 392, 392, 440, 440, 392,
    349, 349, 330, 330, 294, 294, 261,
    392, 392, 349, 349, 330, 330, 294,
    392, 392, 349, 349, 330, 330, 294,
    261, 261, 392, 392, 440, 440, 392,
    349, 349, 330, 330, 294, 294, 261},
    {100, 200, 300, 400, 500, 100, 200,
    300, 400, 500, 100, 200, 300, 400,
    500, 100, 200, 300, 400, 500, 100,
    200, 300, 400, 500, 100, 200, 300,
    400, 500, 100, 200, 300, 400, 500}
};
string name[2] = {
    "Little Star",
    "Test"
};
int noteLength[42] = {
    1, 1, 1, 1, 1, 1, 2,
    1, 1, 1, 1, 1, 1, 2,
    1, 1, 1, 1, 1, 1, 2,
    1, 1, 1, 1, 1, 1, 2,
    1, 1, 1, 1, 1, 1, 2,
    1, 1, 1, 1, 1, 1, 2
};
void PlayNote(int freq)
{
    for(int i = 0; i < kAudioTxBufferSize; i++)
    {
        waveform[i] = (int16_t) (sin((double)i * 2. * M_PI/(double) (kAudioSampleFrequency / freq)) * ((1<<16) - 1));
    }
    audio.spk.play(waveform, kAudioTxBufferSize);
}
void PlaySong(int index)
{
    for(int i = 0; i < 42; i++)
    {
        if(flag == 0) break;
        int length = noteLength[i];
        while(length--)
        {
        // the loop below will play the note for the duration of 1s
        for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j)
        {
            note_queue.call(PlayNote, song[index][i]);
        }
        if(length < 1) wait(1.0);
        }
    }
}
void ButtonEvent()
{   
    if(mode == PLAY) {
        flag = 0;
        mode = MODE_SELECTION;
    }
    else if(mode == MODE_SELECTION) {

    }
    else if(mode == SONG_SELECTION) {

    }
}

int main(void)
{
    note_thread.start(callback(&note_queue, &EventQueue::dispatch_forever));
    button.rise(ButtonEvent);
    mode = PLAY;
    song_iter = 0;
    while(true) {
        if(mode == PLAY) {
            pc.printf("PLAY\r\n");
            uLCD.locate(0, 0);
            uLCD.cls();
            uLCD.printf("Song name : \n");
            uLCD.printf("%s\n", name[song_iter].c_str()); 
            PlaySong(song_iter);
            flag = 1;
            song_iter++;
            song_iter %= NUM_OF_SONGS;
        }
        else if(mode == MODE_SELECTION) {
            audio.spk.pause();
            pc.printf("MODE\r\n");
            uLCD.locate(0, 0);
            uLCD.printf("  Previous Song\n");
            uLCD.printf("  Next Song\n");
            uLCD.printf("  Select Song\n");
            uLCD.locate(0, mode_iter);
            uLCD.printf("->");
            // wait(1.);
            while(!is_changed) {

            }
        }
        else if(mode == SONG_SELECTION) {

        }
    }
}