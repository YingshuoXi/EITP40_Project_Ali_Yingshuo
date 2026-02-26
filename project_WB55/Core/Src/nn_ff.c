/*
 * nn_ff.c
 *
 *  Created on: Feb 22, 2026
 *      Author: yings
 */

#include "nn_ff.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "config.h"

#define HMAX ((NN_FF_H1 > NN_FF_H2 ? NN_FF_H1 : NN_FF_H2) > NN_FF_H3 ? (NN_FF_H1 > NN_FF_H2 ? NN_FF_H1 : NN_FF_H2) : NN_FF_H3)

#define THETA_1 NN_FF_H1
#define THETA_2 NN_FF_H2
#define THETA_3 NN_FF_H3

// weights and bias, stored in RAM
static double W1[NN_FF_H1][NN_FF_IN];
static double b1[NN_FF_H1];

static double W2[NN_FF_H2][NN_FF_H1];
static double b2[NN_FF_H2];

static double W3[NN_FF_H3][NN_FF_H2];
static double b3[NN_FF_H3];



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
void nn_ff_init(void)
{
    const double lim1 = sqrt(6.0 / NN_FF_IN);
    const double lim2 = sqrt(6.0 / NN_FF_H1);
    const double lim3 = sqrt(6.0 / NN_FF_H2);

    for(int i=0; i<NN_FF_H1; i++){			// layer1
        b1[i] = 0.0;
        for(int j=0; j<NN_FF_IN; j++){
            W1[i][j] = kaiming_init(lim1);
        }
    }

    for(int i=0;i<NN_FF_H2;i++){
        b2[i] = 0.0;
        for(int j=0; j<NN_FF_H1; j++){		// layer2
            W2[i][j] = kaiming_init(lim2);
        }
    }

    for(int i=0; i<NN_FF_H3; i++){           // layer3
        b3[i] = 0.0;
        for(int j=0; j<NN_FF_H2; j++){
            W3[i][j] = kaiming_init(lim3);
        }
    }
}

double nn_ff_predict(const double x[NN_FF_IN])
{

    double after_act_1[NN_FF_H1];
	double after_act_2[NN_FF_H2];
	double after_act_3[NN_FF_H3];
	double tmp;

    for(int i=0; i<NN_FF_H1; i++){		// layer1
		tmp = b1[i];
		for(int j=0; j<NN_FF_IN; j++)
			tmp += W1[i][j] * x[j];
		after_act_1[i] = relu(tmp);
    }

    for(int i=0; i<NN_FF_H2; i++){		// layer2
        tmp = b2[i];
        for(int j=0; j<NN_FF_H1; j++)
        	tmp += W2[i][j] * after_act_1[j];
        after_act_2[i] = relu(tmp);
    }
    for(int i=0; i<NN_FF_H3; i++){		// layer3
		tmp = b3[i];
		for(int j=0; j<NN_FF_H2; j++)
			tmp += W3[i][j] * after_act_2[j];
		after_act_3[i] = relu(tmp);
	}

    // Goodness
    double goodness_1 = 0.0, goodness_2 = 0.0, goodness_3 = 0.0;
	for(int i=0; i<NN_FF_H1; i++)		// layer1
		goodness_1 += after_act_1[i] * after_act_1[i];
	for(int i=0; i<NN_FF_H2; i++)		// layer2
		goodness_2 += after_act_2[i] * after_act_2[i];
	for(int i=0; i<NN_FF_H3; i++)	// layer3
		goodness_3 += after_act_3[i] * after_act_3[i];

	// Probability
	double GOODNESS = goodness_1 + goodness_2 + goodness_3;
	double THETA = THETA_1 + THETA_2 + THETA_3;

	return sigmoid(GOODNESS - THETA);
}

double nn_ff_train_one(const double x[NN_FF_IN], int8_t y_int8)
{
    int y_int = (y_int8 != 0) ? 1 : 0;
    double loss_sum = 0.0, dL_dg;


    double before_act[HMAX], after_act[HMAX], input_next[HMAX];
    double tmp;

    // Layer1
	for(int i=0; i<NN_FF_H1; i++){			// Forward
		tmp = b1[i];
		for(int j=0; j<NN_FF_IN; j++)
			tmp += W1[i][j] * x[j];
		before_act[i] = tmp;
		after_act[i] = relu(tmp);
	}

	double goodness_1 = 0.0;			// Goodness
	for(int i=0; i<NN_FF_H1; i++)
		goodness_1 += after_act[i]*after_act[i];

	tmp = y_int ? (THETA_1 - goodness_1) : (goodness_1 - THETA_1);
	double loss_1;
	if(tmp > 11.9 - 6.9) // B-day :)
		loss_1 = tmp;
	else if(tmp < 6.9 - 11.9)
		loss_1 = exp(tmp);
	else loss_1 = log(1.0 + exp(tmp));
	loss_sum += loss_1;

	dL_dg = y_int ? (-sigmoid(THETA_1 - goodness_1)) : (sigmoid(goodness_1 - THETA_1));

    for(int i=0; i<NN_FF_H1; i++){			// Update
        tmp = dL_dg * (2.0 * after_act[i]) * relu_grad(before_act[i]);
        b1[i] -= NN_LR_FF * tmp;
        for(int j=0; j<NN_FF_IN; j++){
            W1[i][j] -= NN_LR_FF * (tmp * x[j]);
        }
    }

    for(int i=0; i<NN_FF_H1; i++){			// Forward again
    	tmp = b1[i];
        for(int j=0; j <NN_FF_IN; j++)
        	tmp += W1[i][j] * x[j];
        input_next[i] = relu(tmp);
    }


    // Layer2
	for(int i=0; i<NN_FF_H2; i++){			// Forward
		tmp = b2[i];
		for(int j=0; j<NN_FF_H1; j++)
			tmp += W2[i][j] * input_next[j];
		before_act[i] = tmp;
		after_act[i] = relu(tmp);
	}

	double goodness_2 = 0.0;			// Goodness
	for(int i=0; i<NN_FF_H2; i++)
		goodness_2 += after_act[i]*after_act[i];

	tmp = y_int ? (THETA_2 - goodness_2) : (goodness_2 - THETA_2);
	double loss_2;
	if(tmp > 11.9 - 6.9)
		loss_2 = tmp;
	else if(tmp < 6.9 - 11.9)
		loss_2 = exp(tmp);
	else loss_2 = log(1.0 + exp(tmp));
	loss_sum += loss_2;

	dL_dg = y_int ? (-sigmoid(THETA_2 - goodness_2)) : (sigmoid(goodness_2 - THETA_2));

	for(int i=0; i<NN_FF_H2; i++){			// Update
		tmp = dL_dg * (2.0 * after_act[i]) * relu_grad(before_act[i]);
		b2[i] -= NN_LR_FF * tmp;
		for(int j=0; j<NN_FF_H1; j++){
			W2[i][j] -= NN_LR_FF * (tmp * input_next[j]);
		}
	}

	for(int i=0; i<NN_FF_H2; i++){			// Forward again
		tmp = b2[i];
		for(int j=0; j <NN_FF_H1; j++)
			tmp += W2[i][j] * input_next[j];
		after_act[i] = relu(tmp);
	}
	for(int i=0; i<NN_FF_H2; i++){
		input_next[i] = after_act[i];
	}

	// Layer3
	for(int i=0; i<NN_FF_H3; i++){			// Forward
		tmp = b3[i];
		for(int j=0; j<NN_FF_H2; j++)
			tmp += W3[i][j] * input_next[j];
		before_act[i] = tmp;
		after_act[i] = relu(tmp);
	}

	double goodness_3 = 0.0;			// Goodness
	for(int i=0; i<NN_FF_H3; i++)
		goodness_3 += after_act[i]*after_act[i];

	tmp = y_int ? (THETA_3 - goodness_3) : (goodness_3 - THETA_3);
	double loss_3;
	if(tmp > 11.9 - 6.9)
		loss_3 = tmp;
	else if(tmp < 6.9 - 11.9)
		loss_3 = exp(tmp);
	else loss_3 = log(1.0 + exp(tmp));
	loss_sum += loss_3;

	dL_dg = y_int ? (-sigmoid(THETA_3 - goodness_3)) : (sigmoid(goodness_3 - THETA_3));

	for(int i=0; i<NN_FF_H3; i++){			// Update
		tmp = dL_dg * (2.0 * after_act[i]) * relu_grad(before_act[i]);
		b3[i] -= NN_LR_FF * tmp;
		for(int j=0; j<NN_FF_H2; j++){
			W3[i][j] -= NN_LR_FF * (tmp * input_next[j]);
		}
	}

	return loss_sum;
}
