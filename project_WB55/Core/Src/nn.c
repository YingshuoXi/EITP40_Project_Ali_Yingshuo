/*
 * nn.c
 *
 *  Created on: Feb 21, 2026
 *      Author: yings
 */

#include "nn.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "config.h"

// weights and bias, stored in RAM
static float W1[NN_H1][NN_IN];
static float b1[NN_H1];

static float W2[NN_H2][NN_H1];
static float b2[NN_H2];

static float W3[NN_H2];
static float b3;

static inline float relu(float x){
	return x > 0.0 ? x : 0.0;
}

static inline float relu_grad(float x){
	return x > 0.0 ? 1.0 : 0.0;
}

// Use numerical stable implementation to avoid overflow
static inline float sigmoid(float z){
    if (z >= 0.0) {
        float ez = exp(-z);
        return 1.0 / (1.0 + ez);
    } else {
        float ez = exp(z);
        return ez / (1.0 + ez);
    }
}

// Random initialization
static inline float kaiming_init(float limit)
{
    float u = (float)rand() / (float)RAND_MAX;  // [0,1]
    return (u * 2.0 - 1.0) * limit;                // [-limit, limit]
}
void nn_init(void)
{
    const float lim1 = sqrt(6.0 / NN_IN);
    const float lim2 = sqrt(6.0 / NN_H1);
    const float lim3 = sqrt(6.0 / NN_H2);

    for(int i=0; i<NN_H1; i++){			// layer1
        b1[i] = 0.0;
        for(int j=0; j<NN_IN; j++){
            W1[i][j] = kaiming_init(lim1);
        }
    }

    for(int i=0;i<NN_H2;i++){
        b2[i] = 0.0;
        for(int j=0; j<NN_H1; j++){		// layer2
            W2[i][j] = kaiming_init(lim2);
        }
    }

    for(int j=0; j<NN_H2; j++){			// layer3
        W3[j] = kaiming_init(lim3);
    }
    b3 = 0.0;
}

float nn_predict(const float x[NN_IN])
{
    float after_act_1[NN_H1];
    float after_act_2[NN_H2];
    float tmp;

    for(int i=0; i<NN_H1; i++){		// layer1
		tmp = b1[i];
		for(int j=0; j<NN_IN; j++)
			tmp += W1[i][j] * x[j];
		after_act_1[i] = relu(tmp);
    }
    for(int i=0; i<NN_H2; i++){		// layer2
        tmp = b2[i];
        for(int j=0; j<NN_H1; j++)
        	tmp += W2[i][j] * after_act_1[j];
        after_act_2[i] = relu(tmp);
    }
    tmp = b3;						// layer3
    for(int j=0; j<NN_H2; j++)
    	tmp += W3[j] * after_act_2[j];
    return sigmoid(tmp);
}

float nn_train_one(const float x[NN_IN], int8_t y_int8)
{
    const float y_float = (y_int8 == 0) ? 0.0 : 1.0;

    // Forward
    float before_act_1[NN_H1], after_act_1[NN_H1];
    float before_act_2[NN_H2], after_act_2[NN_H2];
    float before_act_3, after_act_3;
    float tmp;

    for(int i=0; i<NN_H1; i++){		// layer1
		tmp = b1[i];
		for(int j=0; j<NN_IN; j++)
			tmp += W1[i][j] * x[j];
		before_act_1[i] = tmp;
		after_act_1[i] = relu(tmp);
	}
	for(int i=0; i<NN_H2; i++){		// layer2
		tmp = b2[i];
		for(int j=0; j<NN_H1; j++)
			tmp += W2[i][j] * after_act_1[j];
		before_act_2[i] = tmp;
		after_act_2[i] = relu(tmp);
	}
	tmp = b3;						// layer3
	for(int j=0; j<NN_H2; j++)
		tmp += W3[j] * after_act_2[j];
	before_act_3 = tmp;
    after_act_3 = sigmoid(before_act_3);

    // loss Function
    float eps = 1e-12;
    float loss = -( y_float * log(after_act_3 + eps)
    	+ (1.0 - y_float) * log(1.0 - after_act_3 + eps) );


    // Backpropagation

    // d_Layer3
    float d_before_act_3 = after_act_3 - y_float;
    float d_b3 = d_before_act_3;

    // d_Layer2
    float d_before_act_2[NN_H2];
    for(int i=0; i<NN_H2; i++){
        d_before_act_2[i] = d_before_act_3 * W3[i] * relu_grad(before_act_2[i]);
    }

    // d_Layer1
    float d_before_act_1[NN_H1];
    for(int j=0; j<NN_H1; j++){
        tmp = 0.0;
        for(int i=0; i<NN_H2; i++){
            tmp += W2[i][j] * d_before_act_2[i];
        }
        d_before_act_1[j] = tmp * relu_grad(before_act_1[j]);
    }


    // SGD update

    // Layer3
    for(int j=0; j<NN_H2; j++){
        W3[j] -= NN_LR_BACKPROP * (d_before_act_3 * after_act_2[j]);
    }
    b3 -= NN_LR_BACKPROP * d_b3;

    // Layer2
    for(int i=0; i<NN_H2; i++){
        b2[i] -= NN_LR_BACKPROP * d_before_act_2[i];
        for(int j=0; j<NN_H1; j++){
            W2[i][j] -= NN_LR_BACKPROP * (d_before_act_2[i] * after_act_1[j]);
        }
    }

    // Layer1
    for(int i=0; i<NN_H1; i++){
        b1[i] -= NN_LR_BACKPROP * d_before_act_1[i];
        for(int j=0; j<NN_IN; j++){
            W1[i][j] -= NN_LR_BACKPROP * (d_before_act_1[i] * x[j]);
        }
    }

    return loss;
}

size_t nn_state_size(void)
{
    return sizeof(W1) + sizeof(b1) +
           sizeof(W2) + sizeof(b2) +
           sizeof(W3) + sizeof(b3);
}

void nn_state_export(void *dst)
{
    uint8_t *p = (uint8_t *)dst;

    memcpy(p, W1, sizeof(W1)); p += sizeof(W1);
    memcpy(p, b1, sizeof(b1)); p += sizeof(b1);

    memcpy(p, W2, sizeof(W2)); p += sizeof(W2);
    memcpy(p, b2, sizeof(b2)); p += sizeof(b2);

    memcpy(p, W3, sizeof(W3)); p += sizeof(W3);
    memcpy(p, &b3, sizeof(b3));
}

uint8_t nn_state_import(const void *src, size_t len)
{
    if (len != nn_state_size()) {
        return 0;
    }

    const uint8_t *p = (const uint8_t *)src;

    memcpy(W1, p, sizeof(W1)); p += sizeof(W1);
    memcpy(b1, p, sizeof(b1)); p += sizeof(b1);

    memcpy(W2, p, sizeof(W2)); p += sizeof(W2);
    memcpy(b2, p, sizeof(b2)); p += sizeof(b2);

    memcpy(W3, p, sizeof(W3)); p += sizeof(W3);
    memcpy(&b3, p, sizeof(b3));

    return 1;
}
