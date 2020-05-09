#include "mbed.h"
#include <cmath>
#include <string>
#include <iostream>
#include "DA7212.h"
#include "uLCD_4DGL.h"
#include "accelerometer_handler.h"
#include "config.h"
#include "magic_wand_model_data.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#define NUM_OF_SONGS 2
#define NUM_OF_MODES 3
#define NUM_OF_FREQUNCYS 42
enum Mode {
    PLAY, MODE_SELECTION, SONG_SELECTION
};
DA7212 audio;
Serial pc(USBTX, USBRX);
int16_t waveform[kAudioTxBufferSize];
uLCD_4DGL uLCD(D1, D0, D2);
InterruptIn button(SW2);
int id;
int flag = 1;
int song_iter = 0;
int mode_iter = 0;
int choose_iter = 0;
Mode mode;
bool exit_DNN = false;
Thread note_thread(osPriorityNormal);
Thread DNN_thread(osPriorityNormal, 120*1024);
EventQueue note_queue(32 * EVENTS_EVENT_SIZE);
EventQueue DNN_queue(32 * EVENTS_EVENT_SIZE);
DigitalOut green_led(LED2);

// DNN variable
constexpr int kTensorArenaSize = 50 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
bool should_clear_buffer = false;
bool got_data = false;
int gesture_index;
static tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
int input_length;
TfLiteTensor* model_input;
tflite::MicroInterpreter* interpreter;


// Return the result of the last prediction
int PredictGesture(float* output) {
    // How many times the most recent gesture has been matched in a row
    static int continuous_count = 0;
    // The result of the last prediction
    static int last_predict = -1;

    // Find whichever output has a probability > 0.8 (they sum to 1)
    int this_predict = -1;
    for (int i = 0; i < label_num; i++) {
        if (output[i] > 0.8) this_predict = i;
    }

    // No gesture was detected above the threshold
    if (this_predict == -1) {
        continuous_count = 0;
        last_predict = label_num;
        return label_num;
    }

    if (last_predict == this_predict) {
        continuous_count += 1;
    } else {
        continuous_count = 0;
    }
    last_predict = this_predict;

    // If we haven't yet had enough consecutive matches for this gesture,
    // report a negative result
    if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {
        return label_num;
    }
    // Otherwise, we've seen a positive result, so clear all our variables
    // and report it
    continuous_count = 0;
    last_predict = -1;

    return this_predict;
}

int song[NUM_OF_SONGS][NUM_OF_FREQUNCYS];
string name[NUM_OF_SONGS] = {
    "Little Star",
    "Test"
};
int noteLength[NUM_OF_FREQUNCYS] = {
    1, 1, 1, 1, 1, 1, 2,
    1, 1, 1, 1, 1, 1, 2,
    1, 1, 1, 1, 1, 1, 2,
    1, 1, 1, 1, 1, 1, 2,
    1, 1, 1, 1, 1, 1, 2,
    1, 1, 1, 1, 1, 1, 2
};
void LoadMusic()
{
    green_led = 0;
    int i = 0;
    int serial_count = 0;
    song_iter = 0;
    char serial_buffer[32];
    while(i < NUM_OF_FREQUNCYS * NUM_OF_SONGS)
    {
        if(pc.readable())
        {
            serial_buffer[serial_count] = pc.getc();
            serial_count++;
            if(serial_count == 5)
            {
                serial_buffer[serial_count] = '\0';
                song[song_iter][i%NUM_OF_FREQUNCYS] = (int) (atof(serial_buffer) * 1000.0);
                serial_count = 0;
                i++;
                if(i % NUM_OF_FREQUNCYS == 0) {
                    song_iter ++;
                }
            }
        }
    }
    green_led = 1;
}
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
    for(int i = 0; i < NUM_OF_FREQUNCYS; i++)
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
        // audio.spk.status(false);
        flag = 0;
        mode = MODE_SELECTION;
        mode_iter = 0;
    }
    else if(mode == MODE_SELECTION) {
        // audio.spk.status(true);
        exit_DNN = true;
        if(mode_iter == 0) {
            mode = PLAY;
            song_iter--;
            if(song_iter < 0) song_iter = NUM_OF_SONGS - 1;
        }
        else if(mode_iter == 1) {
            mode = PLAY;
            song_iter++;
            song_iter %= NUM_OF_SONGS;
        }
        else if(mode_iter == 2) {
            mode = SONG_SELECTION;
            choose_iter = 0;
        }
    }
    else if(mode == SONG_SELECTION) {
        song_iter = choose_iter;
        exit_DNN = true;
        mode = PLAY;
    }
}
int DNN() {
    // error_reporter->Report("Set up successful...\n");
    while (!exit_DNN) {
        // Attempt to read new data from the accelerometer
        got_data = ReadAccelerometer(error_reporter, model_input->data.f,
                                    input_length, should_clear_buffer);

        // If there was no new data,
        // don't try to clear the buffer again and wait until next time
        if (!got_data) {
            should_clear_buffer = false;
            continue;
        }

        // Run inference, and report any error
        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
            error_reporter->Report("Invoke failed on index: %d\n", begin_index);
            continue;
        }

        // Analyze the results to obtain a prediction
        gesture_index = PredictGesture(interpreter->output(0)->data.f);

        // Clear the buffer next time we read data
        should_clear_buffer = gesture_index < label_num;

        // Produce an output
        if (gesture_index < label_num) {
            error_reporter->Report(config.output_message[gesture_index]);
            if(mode == MODE_SELECTION) {
                // ring gesture
                if(gesture_index == 0) {
                    mode_iter--;
                    if(mode_iter < 0) mode_iter = NUM_OF_MODES - 1;
                    break;
                }
                // slope gesture
                else if(gesture_index == 1) {
                    mode_iter++;
                    mode_iter %= NUM_OF_MODES;
                    break;
                }
            }
            if(mode == SONG_SELECTION) {
                // ring gesture
                if(gesture_index == 0) {
                    choose_iter--;
                    if(choose_iter < 0) choose_iter = NUM_OF_MODES - 1;
                    break;
                }
                // slope gesture
                else if(gesture_index == 1) {
                    choose_iter++;
                    choose_iter %= NUM_OF_MODES;
                    break;
                }
            }
        }
    }
}
void Music() {
    song_iter = 0;
    while(true) {
        if(mode == PLAY) {
            // audio.spk.play();
            uLCD.locate(0, 0);
            uLCD.cls();
            uLCD.printf("Song name : \n");
            uLCD.printf("%s\n", name[song_iter].c_str()); 
            PlaySong(song_iter);
            if(flag == 1) {
                song_iter++;
                song_iter %= NUM_OF_SONGS;
            }
            flag = 1;
        }
        else if(mode == MODE_SELECTION) {
            // audio.spk.pause();
            uLCD.locate(0, 0);
            uLCD.cls();
            uLCD.printf("  Previous Song\n");
            uLCD.printf("  Next Song\n");
            uLCD.printf("  Select Song\n");
            uLCD.locate(0, mode_iter);
            uLCD.printf("->");
            DNN();
            exit_DNN = false;
        }
        else if(mode == SONG_SELECTION) {
            uLCD.locate(0, 0);
            uLCD.cls();
            for(int i = 0; i < NUM_OF_SONGS; i++) {
                uLCD.printf("  %s\n", name[i].c_str());
            }
            uLCD.locate(0, choose_iter);
            uLCD.printf("->");
            DNN();
            exit_DNN = false;
        }
    }
}
int main(void)
{
    note_thread.start(callback(&note_queue, &EventQueue::dispatch_forever));
    button.rise(ButtonEvent);
    mode = PLAY;
    song_iter = 0;
    // set up DNN
    const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        error_reporter->Report(
            "Model provided is schema version %d not equal "
            "to supported version %d.",
            model->version(), TFLITE_SCHEMA_VERSION);
        return -1;
    }
    static tflite::MicroOpResolver<6> micro_op_resolver;
    micro_op_resolver.AddBuiltin(
        tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
        tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
                                tflite::ops::micro::Register_MAX_POOL_2D());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                                tflite::ops::micro::Register_CONV_2D());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                                tflite::ops::micro::Register_FULLY_CONNECTED());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                                tflite::ops::micro::Register_SOFTMAX());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                                tflite::ops::micro::Register_RESHAPE(), 1);
    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    interpreter->AllocateTensors();
    model_input = interpreter->input(0);
    if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
        (model_input->dims->data[1] != config.seq_length) ||
        (model_input->dims->data[2] != kChannelNumber) ||
        (model_input->type != kTfLiteFloat32)) {
        error_reporter->Report("Bad input tensor parameters in model");
        return -1;
    }

    input_length = model_input->bytes / sizeof(float);

    TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
    if (setup_status != kTfLiteOk) {
        error_reporter->Report("Set up failed\n");
        return -1;
    }
    LoadMusic();
    Music();
}



