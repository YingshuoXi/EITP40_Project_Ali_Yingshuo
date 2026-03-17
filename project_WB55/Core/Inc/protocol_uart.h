/*
 * protocol_uart.h
 *
 *  Created on: Feb 19, 2026
 *      Author: yings
 */

#pragma once
#include "stm32wbxx_hal.h"
#include <stdint.h>

void protocol_init(UART_HandleTypeDef *hlpuart);	// Initialization
void protocol_start_uart_rx(void);				// Ready to receive data
void protocol_uart_rx_byte(uint8_t b);      	// Message analysis
void protocol_while(void);  					// Used in the while loop
void protocol_set_idle(uint8_t idle);			// Set Idle(1)
void protocol_resume_requesting(void);  		// Request next sample when training is done

void protocol_send_req(void);              		// Send request
void protocol_send_end(void);					// Stop Python code
void protocol_send_results(float loss, float p, uint8_t correct);	// Send Results
void protocol_send_infer_req(void);				// Send request for inference
void protocol_send_inference_acc(float acc);	// Send inference acc
uint8_t protocol_is_inference_mode(void);		// Inference mode setting

uint8_t protocol_is_train_finished(void);
void protocol_clear_train_finished(void);
uint8_t protocol_is_infer_finished(void);
void protocol_clear_infer_finished(void);
void protocol_after_infer_processed(void);

void protocol_send_test_req(void);
void protocol_send_test_prediction(float probability, uint8_t pred);
uint8_t protocol_is_test_mode(void);

uint8_t protocol_is_test_finished(void);
void protocol_clear_test_finished(void);
void protocol_after_test_processed(void);
