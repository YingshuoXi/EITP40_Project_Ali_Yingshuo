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

#define IN  50
#define H1  20
#define H2  10
//#define OUT 1

static double global_lr = 1e-3;

// weights and bias, stored in RAM
static double W1[H1][IN];
static double b1[H1];

static double W2[H2][H1];
static double b2[H2];

static double W3[H2];
static double b3;

static inline double relu(double x){
	return x > 0.0 ? x : 0.0;
}

static inline double relu_grad(double x){
	return x > 0.0 ? 1.0 : 0.0;
}

// Use numerical stable implementation to avoid overflow
static inline double sigmoid(double z){
    if (z >= 0.0) {
        double ez = exp(-z);
        return 1.0 / (1.0 + ez);
    } else {
        double ez = exp(z);
        return ez / (1.0 + ez);
    }
}

// Random initialization
static inline double kaiming_init(double limit)
{
    double u = (double)rand() / (double)RAND_MAX;  // [0,1]
    return (u * 2.0 - 1.0) * limit;                // [-limit, limit]
}
void nn_init(double lr)
{
    global_lr = lr;

    const double lim1 = sqrt(6.0 / IN);
    const double lim2 = sqrt(6.0 / H1);
    const double lim3 = sqrt(6.0 / H2);

    for(int i=0;i<H1;i++){			// layer1
        b1[i] = 0.0;
        for(int j=0;j<IN;j++){
            W1[i][j] = kaiming_init(lim1);
        }
    }

    for(int i=0;i<H2;i++){
        b2[i] = 0.0;
        for(int j=0;j<H1;j++){		// layer2
            W2[i][j] = kaiming_init(lim2);
        }
    }

    for(int j=0;j<H2;j++){			// layer3
        W3[j] = kaiming_init(lim3);
    }
    b3 = 0.0;
}

double nn_predict(const double x[50])
{
    double after_act_1[H1];
    double after_act_2[H2];
    double tmp;

    for(int i=0; i<H1; i++){		// layer1
		tmp = b1[i];
		for(int j=0; j<IN; j++)
			tmp += W1[i][j] * x[j];
		after_act_1[i] = relu(tmp);
    }
    for(int i=0; i<H2; i++){		// layer2
        tmp = b2[i];
        for(int j=0; j<H1; j++)
        	tmp += W2[i][j] * after_act_1[j];
        after_act_2[i] = relu(tmp);
    }
    tmp = b3;						// layer3
    for(int j=0; j<H2; j++)
    	tmp += W3[j] * after_act_2[j];
    return sigmoid(tmp);
}

double nn_train_one(const double x[50], int8_t y_int8)
{
    const double y_double = (y_int8 == 0) ? 0.0 : 1.0;

    // Forward
    double before_act_1[H1], after_act_1[H1];
    double before_act_2[H2], after_act_2[H2];
    double before_act_3, after_act_3;
    double tmp;

    for(int i=0; i<H1; i++){		// layer1
		tmp = b1[i];
		for(int j=0; j<IN; j++)
			tmp += W1[i][j] * x[j];
		before_act_1[i] = tmp;
		after_act_1[i] = relu(tmp);
	}
	for(int i=0; i<H2; i++){		// layer2
		tmp = b2[i];
		for(int j=0; j<H1; j++)
			tmp += W2[i][j] * after_act_1[j];
		before_act_2[i] = tmp;
		after_act_2[i] = relu(tmp);
	}
	tmp = b3;						// layer3
	for(int j=0; j<H2; j++)
		tmp += W3[j] * after_act_2[j];
	before_act_3 = tmp;
    after_act_3 = sigmoid(before_act_3);

    // loss Function
    double eps = 1e-12;
    double loss = -( y_double * log(after_act_3 + eps)
    	+ (1.0 - y_double) * log(1.0 - after_act_3 + eps) );


    // Backpropagation

    // d_Layer3
    double d_before_act_3 = after_act_3 - y_double;
    double d_b3 = d_before_act_3;

    // d_Layer2
    double d_before_act_2[H2];
    for(int i=0; i<H2; i++){
        d_before_act_2[i] = d_before_act_3 * W3[i] * relu_grad(before_act_2[i]);
    }

    // d_Layer1
    double d_before_act_1[H1];
    for(int j=0; j<H1; j++){
        tmp = 0.0;
        for(int i=0; i<H2; i++){
            tmp += W2[i][j] * d_before_act_2[i];
        }
        d_before_act_1[j] = tmp * relu_grad(before_act_1[j]);
    }


    // SGD update

    // Layer3
    for(int j=0; j<H2; j++){
        W3[j] -= global_lr * (d_before_act_3 * after_act_2[j]);
    }
    b3 -= global_lr * d_b3;

    // Layer2
    for(int i=0; i<H2; i++){
        b2[i] -= global_lr * d_before_act_2[i];
        for(int j=0; j<H1; j++){
            W2[i][j] -= global_lr * (d_before_act_2[i] * after_act_1[j]);
        }
    }

    // Layer1
    for(int i=0; i<H1; i++){
        b1[i] -= global_lr * d_before_act_1[i];
        for(int j=0; j<IN; j++){
            W1[i][j] -= global_lr * (d_before_act_1[i] * x[j]);
        }
    }

    return loss;
}
