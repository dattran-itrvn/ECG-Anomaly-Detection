/*******************************************************************************
* Copyright (C) 2019-2023 Maxim Integrated Products, Inc., All rights Reserved.
*
* This software is protected by copyright laws of the United States and
* of foreign countries. This material may also be protected by patent laws
* and technology transfer regulations of the United States and of foreign
* countries. This software is furnished under a license agreement and/or a
* nondisclosure agreement and may only be used or reproduced in accordance
* with the terms of those agreements. Dissemination of this information to
* any party or parties not specified in the license agreement and/or
* nondisclosure agreement is expressly prohibited.
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
* OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL MAXIM INTEGRATED BE LIABLE FOR ANY CLAIM, DAMAGES
* OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
* OTHER DEALINGS IN THE SOFTWARE.
*
* Except as contained in this notice, the name of Maxim Integrated
* Products, Inc. shall not be used except as stated in the Maxim Integrated
* Products, Inc. Branding Policy.
*
* The mere transfer of this software does not imply any licenses
* of trade secrets, proprietary technology, copyrights, patents,
* trademarks, maskwork rights, or any other form of intellectual
* property whatsoever. Maxim Integrated Products, Inc. retains all
* ownership rights.
*******************************************************************************/

// autoencoder_ecg
// This file was @generated by ai8xize.py --test-dir sdk/Examples/MAX78000/CNN --prefix autoencoder_ecg --checkpoint-file trained/ai85-autoencoder-ecg-qat-q.pth.tar --config-file networks/ai85-autoencoder-ecg.yaml --sample-input tests/sample_sampleecg_forevalwithsignal.npy --energy --device MAX78000 --timer 0 --display-checkpoint --verbose

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "mxc.h"
#include "cnn.h"
#include "sampledata.h"
#include "sampleoutput.h"

volatile uint32_t cnn_time; // Stopwatch

void fail(void)
{
  printf("\n*** FAIL ***\n\n");
  while (1);
}

// 188-channel 1x1 data input (188 bytes total / 1 bytes per channel):
// HWC 1x1, channels 0 to 3
// HWC 1x1, channels 64 to 67
// HWC 1x1, channels 128 to 131
static const uint32_t input_0[] = SAMPLE_INPUT_0;

// HWC 1x1, channels 4 to 7
// HWC 1x1, channels 68 to 71
// HWC 1x1, channels 132 to 135
static const uint32_t input_4[] = SAMPLE_INPUT_4;

// HWC 1x1, channels 8 to 11
// HWC 1x1, channels 72 to 75
// HWC 1x1, channels 136 to 139
static const uint32_t input_8[] = SAMPLE_INPUT_8;

// HWC 1x1, channels 12 to 15
// HWC 1x1, channels 76 to 79
// HWC 1x1, channels 140 to 143
static const uint32_t input_12[] = SAMPLE_INPUT_12;

// HWC 1x1, channels 16 to 19
// HWC 1x1, channels 80 to 83
// HWC 1x1, channels 144 to 147
static const uint32_t input_16[] = SAMPLE_INPUT_16;

// HWC 1x1, channels 20 to 23
// HWC 1x1, channels 84 to 87
// HWC 1x1, channels 148 to 151
static const uint32_t input_20[] = SAMPLE_INPUT_20;

// HWC 1x1, channels 24 to 27
// HWC 1x1, channels 88 to 91
// HWC 1x1, channels 152 to 155
static const uint32_t input_24[] = SAMPLE_INPUT_24;

// HWC 1x1, channels 28 to 31
// HWC 1x1, channels 92 to 95
// HWC 1x1, channels 156 to 159
static const uint32_t input_28[] = SAMPLE_INPUT_28;

// HWC 1x1, channels 32 to 35
// HWC 1x1, channels 96 to 99
// HWC 1x1, channels 160 to 163
static const uint32_t input_32[] = SAMPLE_INPUT_32;

// HWC 1x1, channels 36 to 39
// HWC 1x1, channels 100 to 103
// HWC 1x1, channels 164 to 167
static const uint32_t input_36[] = SAMPLE_INPUT_36;

// HWC 1x1, channels 40 to 43
// HWC 1x1, channels 104 to 107
// HWC 1x1, channels 168 to 171
static const uint32_t input_40[] = SAMPLE_INPUT_40;

// HWC 1x1, channels 44 to 47
// HWC 1x1, channels 108 to 111
// HWC 1x1, channels 172 to 175
static const uint32_t input_44[] = SAMPLE_INPUT_44;

// HWC 1x1, channels 48 to 51
// HWC 1x1, channels 112 to 115
// HWC 1x1, channels 176 to 179
static const uint32_t input_48[] = SAMPLE_INPUT_48;

// HWC 1x1, channels 52 to 55
// HWC 1x1, channels 116 to 119
// HWC 1x1, channels 180 to 183
static const uint32_t input_52[] = SAMPLE_INPUT_52;

// HWC 1x1, channels 56 to 59
// HWC 1x1, channels 120 to 123
// HWC 1x1, channels 184 to 187
static const uint32_t input_56[] = SAMPLE_INPUT_56;

// HWC 1x1, channels 60 to 63
// HWC 1x1, channels 124 to 127
// HWC 1x1, channels 188 to 191
static const uint32_t input_60[] = SAMPLE_INPUT_60;

void load_input(void)
{
  // This function loads the sample data input -- replace with actual data

  memcpy32((uint32_t *) 0x50400000, input_0, 3);
  memcpy32((uint32_t *) 0x50408000, input_4, 3);
  memcpy32((uint32_t *) 0x50410000, input_8, 3);
  memcpy32((uint32_t *) 0x50418000, input_12, 3);
  memcpy32((uint32_t *) 0x50800000, input_16, 3);
  memcpy32((uint32_t *) 0x50808000, input_20, 3);
  memcpy32((uint32_t *) 0x50810000, input_24, 3);
  memcpy32((uint32_t *) 0x50818000, input_28, 3);
  memcpy32((uint32_t *) 0x50c00000, input_32, 3);
  memcpy32((uint32_t *) 0x50c08000, input_36, 3);
  memcpy32((uint32_t *) 0x50c10000, input_40, 3);
  memcpy32((uint32_t *) 0x50c18000, input_44, 3);
  memcpy32((uint32_t *) 0x51000000, input_48, 3);
  memcpy32((uint32_t *) 0x51008000, input_52, 3);
  memcpy32((uint32_t *) 0x51010000, input_56, 3);
  memcpy32((uint32_t *) 0x51018000, input_60, 3);
}

// Expected output of layer 6 for autoencoder_ecg given the sample input (known-answer test)
// Delete this function for production code
static const uint32_t sample_output[] = SAMPLE_OUTPUT;
int check_output(void)
{
  int i;
  uint32_t mask, len;
  volatile uint32_t *addr;
  const uint32_t *ptr = sample_output;

  while ((addr = (volatile uint32_t *) *ptr++) != 0) {
    mask = *ptr++;
    len = *ptr++;
    for (i = 0; i < len; i++)
      if ((*addr++ & mask) != *ptr++) {
        printf("Data mismatch (%d/%d) at address 0x%08x: Expected 0x%08x, read 0x%08x.\n",
               i + 1, len, addr - 1, *(ptr - 1), *(addr - 1) & mask);
        return CNN_FAIL;
      }
  }

  return CNN_OK;
}

static int32_t ml_data32[(CNN_NUM_OUTPUTS + 3) / 4];

int main(void)
{
  int i;
  MXC_ICC_Enable(MXC_ICC0); // Enable cache

  // Switch to 100 MHz clock
  MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
  SystemCoreClockUpdate();

  printf("Waiting...\n");

  // DO NOT DELETE THIS LINE:
  MXC_Delay(SEC(2)); // Let debugger interrupt if needed

  cnn_disable(); // Disable clock and power to CNN
  // Enable primary clock
  MXC_SYS_ClockSourceEnable(MXC_SYS_CLOCK_IPO);

  printf("Measuring system base (idle) power...\n");
  SYS_START;
  MXC_Delay(SEC(1));
  SYS_COMPLETE;

  // Enable peripheral, enable CNN interrupt, turn on CNN clock
  // CNN clock: APB (50 MHz) div 1
  cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);

  printf("\n*** CNN Inference Test autoencoder_ecg ***\n");

  cnn_init(); // Bring state machine into consistent state

  printf("Measuring weight loading...\n");
  CNN_START;
  for (i = 0; i < 100; i++)
    cnn_load_weights(); // Load kernels
  CNN_COMPLETE;

  MXC_TMR_Delay(MXC_TMR0, 500000);
  printf("Measuring input loading...\n");
  CNN_START;
  for (i = 0; i < 100; i++)
    load_input(); // Load data input
  CNN_COMPLETE;

  cnn_load_bias();
  cnn_configure(); // Configure state machine

  MXC_TMR_Delay(MXC_TMR0, 500000);
  printf("Measuring input load + inference...\n");
  CNN_START; // Allow capture of processing time
  for (i = 0; i < 100; i++) {
    load_input(); // Load data input
    cnn_start(); // Run inference
    while (cnn_time == 0)
      MXC_LP_EnterSleepMode(); // Wait for CNN
  }
  CNN_COMPLETE;

  if (check_output() != CNN_OK) fail();
  cnn_unload((uint32_t *) ml_data32);

  printf("\n*** PASS ***\n\n");

#ifdef CNN_INFERENCE_TIMER
  printf("Approximate inference time: %u us\n\n", cnn_time);
#endif

  printf("See monitor display for inference energy.\n\n");

  cnn_disable(); // Shut down CNN clock, disable peripheral


  return 0;
}

/*
  SUMMARY OF OPS
  Hardware: 56,036 ops (55,680 macc; 356 comp; 0 add; 0 mul; 0 bitwise)
    Layer 0: 24,192 ops (24,064 macc; 128 comp; 0 add; 0 mul; 0 bitwise)
    Layer 1: 8,256 ops (8,192 macc; 64 comp; 0 add; 0 mul; 0 bitwise)
    Layer 2: 2,080 ops (2,048 macc; 32 comp; 0 add; 0 mul; 0 bitwise)
    Layer 3: 132 ops (128 macc; 4 comp; 0 add; 0 mul; 0 bitwise)
    Layer 4: 160 ops (128 macc; 32 comp; 0 add; 0 mul; 0 bitwise)
    Layer 5: 3,168 ops (3,072 macc; 96 comp; 0 add; 0 mul; 0 bitwise)
    Layer 6: 18,048 ops (18,048 macc; 0 comp; 0 add; 0 mul; 0 bitwise)

  RESOURCE USAGE
  Weight memory: 55,680 bytes out of 442,368 bytes total (12.6%)
  Bias memory:   352 bytes out of 2,048 bytes total (17.2%)
*/

