/*
 * protocol_uart.c
 *
 *  Created on: Feb 19, 2026
 *      Author: yings
 */

#include <save.h>
#include "protocol_uart.h"
#include "stm32wbxx_nucleo.h"
#include <string.h>

#define CODE_START_OF_FRAME_0 0xAA
#define CODE_START_OF_FRAME_1 0x55

#define CODE_REQUEST_DATA      	0x01
#define CODE_DATA              	0x02
#define CODE_ACKNOWLEDGEMENT   	0x03
#define CODE_FINISH 			0x04
#define CODE_END    			0x05

#define EXPECTED_LENGTH (400 + 1)   	// label 1 byte + 50 doubles * 8 bytes

// Waiting time before sending new request
#define REQ_WAITING_PERIOD 20u

static UART_HandleTypeDef *global_hlpuart;
static uint8_t global_rx_byte;

static volatile uint8_t global_flag_pending = 0;  			// 1: something is waiting to be done
static volatile uint8_t global_acknowledgement_status = 0;	// 0: everything is good before acknowledgement; 1: there is an error
static volatile uint16_t global_sequence_number = 0;		// request sequence
static volatile uint16_t global_ack_sequence_number = 0; 	// acknowledgement sequence
static volatile uint32_t global_last_req_tick = 0;
static volatile uint8_t	protocol_idle = 0;					// 1: idle

// FSM by byte: Start of frame - type - sequence - length - data - crc
typedef enum {
    STATE_CODE_START_OF_FRAME_0, STATE_CODE_START_OF_FRAME_1,
    STATE_DATA_TYPE,
    STATE_SEQUENCE_0, STATE_SEQUENCE_1,
    STATE_LENGTH_0, STATE_LENGTH_1,
    STATE_DATA,
    STATE_CRC_0, STATE_CRC_1
} st_t;

static st_t state = STATE_CODE_START_OF_FRAME_0;

static uint8_t  data_type;
static uint16_t message_sequence;
static uint16_t message_length;
static uint16_t message_counter;
static uint8_t  label;
static uint8_t  data[EXPECTED_LENGTH];
static uint16_t crc;

static void handle_message(void);
static void send_frame(uint8_t type, uint16_t seq, const uint8_t *dt, uint16_t len);

// CCITT 16-bit crc check
static uint16_t crc16_ccitt_update(uint16_t c, const uint8_t *data, uint16_t len)
{
    for (uint16_t i = 0; i < len; i++){
        c ^= (uint16_t)data[i] << 8;
        for (uint8_t b = 0; b < 8; b++){
            if (c & 0x8000)
                c = (uint16_t)((c << 1) ^ 0x1021);
            else
                c = (uint16_t)(c << 1);
        }
    }
    return c;
}

// Initialization
void protocol_init(UART_HandleTypeDef *hlpuart)
{
    global_hlpuart = hlpuart;
    state = STATE_CODE_START_OF_FRAME_0;
    global_last_req_tick = HAL_GetTick();
}

// Ready to receive data
void protocol_start_uart_rx(void)
{
    HAL_UART_Receive_IT(global_hlpuart, &global_rx_byte, 1);
}

// LPUART Interrupt function
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *hlpuart)
{
    if (hlpuart == global_hlpuart)
    {
        protocol_uart_rx_byte(global_rx_byte);
        HAL_UART_Receive_IT(global_hlpuart, &global_rx_byte, 1);
    }
}

// Message analysis
void protocol_uart_rx_byte(uint8_t message_byte)
{
    switch (state)
    {
    case STATE_CODE_START_OF_FRAME_0:
        state = (message_byte == CODE_START_OF_FRAME_0) ? STATE_CODE_START_OF_FRAME_1 : STATE_CODE_START_OF_FRAME_0;
        break;

    case STATE_CODE_START_OF_FRAME_1:
        state = (message_byte == CODE_START_OF_FRAME_1) ? STATE_DATA_TYPE : STATE_CODE_START_OF_FRAME_0;
        break;

    case STATE_DATA_TYPE:
        data_type = message_byte;
        state = STATE_SEQUENCE_0;
        break;

    case STATE_SEQUENCE_0:
        message_sequence = message_byte;
        state = STATE_SEQUENCE_1;
        break;

    case STATE_SEQUENCE_1:
        message_sequence |= ((uint16_t)message_byte << 8);
        state = STATE_LENGTH_0;
        break;

    case STATE_LENGTH_0:
        message_length = message_byte;
        state = STATE_LENGTH_1;
        break;

    case STATE_LENGTH_1:
        message_length |= ((uint16_t)message_byte << 8);
        message_counter = 0;

        // If the message is longer than the data buffer, throw it away.
        if (message_length > sizeof(data)) {
            state = STATE_CODE_START_OF_FRAME_0;
        } else {
            state = (message_length == 0) ? STATE_CRC_0 : STATE_DATA;
        }
        break;

    case STATE_DATA:
        data[message_counter++] = message_byte;
        if (message_counter >= message_length)
            state = STATE_CRC_0;
        break;

    case STATE_CRC_0:
        crc = message_byte;
        state = STATE_CRC_1;
        break;

    case STATE_CRC_1:
        crc |= ((uint16_t)message_byte << 8);
        handle_message();
        state = STATE_CODE_START_OF_FRAME_0;
        break;
    }
}

static void handle_message(void)
{
    uint8_t header_without_sof[5];
    header_without_sof[0] = data_type;
    header_without_sof[1] = (uint8_t)(message_sequence & 0xFF);
    header_without_sof[2] = (uint8_t)(message_sequence >> 8);
    header_without_sof[3] = (uint8_t)(message_length & 0xFF);
    header_without_sof[4] = (uint8_t)(message_length >> 8);

    // calculate crc
    uint16_t crc_calculated = 0xFFFF;
    crc_calculated = crc16_ccitt_update(crc_calculated, &header_without_sof, 5);
    if (message_length && data) {
    	crc_calculated = crc16_ccitt_update(crc_calculated, data, message_length);
	}

    // crc check, if fail: re-send request
    if (crc_calculated != crc)
    {
    	global_acknowledgement_status = 1;
		global_ack_sequence_number = message_sequence;
		global_flag_pending = 1;
        return;
    }

    if (data_type == CODE_FINISH)
    {
    	protocol_set_idle(1);
        return;
    }

    // Data
    if (data_type != CODE_DATA)
        return;

    // Length check, if fail: re-send request
    if (message_length != EXPECTED_LENGTH)
    {
    	global_acknowledgement_status = 1;
		global_ack_sequence_number = message_sequence;
		global_flag_pending = 1;
        return;
    }

    // Sequence check
    if (message_sequence == global_sequence_number)		// correct data
    {
        // Expected new packet
    	label = data[0];
        save_store_vector(&data[1], EXPECTED_LENGTH - 1);
        HAL_GPIO_TogglePin(GPIOB, GPIO_PIN_5);

        global_acknowledgement_status = 0;
        global_ack_sequence_number = global_sequence_number;
        global_flag_pending = 1;
        return;
    }

    // old package, send acknowledgement again but not store it.
    if (message_sequence < global_sequence_number)
    {
        global_acknowledgement_status = 0;
        global_ack_sequence_number = message_sequence;
        global_flag_pending = 1;
        return;
    }

    // Other cases
    global_acknowledgement_status = 1;
    global_ack_sequence_number = message_sequence;
    global_flag_pending = 1;
}

// send message: start of frame - type - sequence - sequence - data - crc
static void send_frame(uint8_t type, uint16_t sequence, const uint8_t *data, uint16_t length)
{
    uint8_t header[7];
    header[0] = CODE_START_OF_FRAME_0;
    header[1] = CODE_START_OF_FRAME_1;
    header[2] = type;
    header[3] = (uint8_t)(sequence & 0xFF);
    header[4] = (uint8_t)(sequence >> 8);
    header[5] = (uint8_t)(length & 0xFF);
    header[6] = (uint8_t)(length >> 8);

    uint16_t crc = 0xFFFF;
    crc = crc16_ccitt_update(crc, &header[2], 5);
    if (length && data) {
    	crc = crc16_ccitt_update(crc, data, length);
	}
    uint8_t crcs[2] = { (uint8_t)(crc & 0xFF), (uint8_t)(crc >> 8) };

    HAL_UART_Transmit(global_hlpuart, header, sizeof(header), 1000);
    if (length && data)
        HAL_UART_Transmit(global_hlpuart, (uint8_t*)data, length, 1000);
    HAL_UART_Transmit(global_hlpuart, crcs, 2, 1000);
}

// Send request (REQ(expected_seq))
void protocol_send_req(void)
{
    send_frame(CODE_REQUEST_DATA, global_sequence_number, NULL, 0);
    global_last_req_tick = HAL_GetTick();
}

// Idle state: without doing anything
void protocol_set_idle(uint8_t idle)
{
    protocol_idle = idle ? 1 : 0;
    if (protocol_idle)
    {
        global_flag_pending = 0;
        BSP_LED_On(LED_GREEN);
    } else{
    	BSP_LED_Off(LED_GREEN);
    }
}

// Shut down Python
void protocol_send_end(void)
{
    send_frame(CODE_END, global_sequence_number, NULL, 0);
}

// Used in the while loop (call frequently)
void protocol_while(void)
{
	// Idle: do nothing
	if (protocol_idle)
	{
		global_flag_pending = 0;
		return;
	}

	// Periodically Re-send Request if the data does not arrive
    uint32_t now = HAL_GetTick();
    if ((uint32_t)(now - global_last_req_tick) >= REQ_WAITING_PERIOD)
    {
        protocol_send_req();
    }

    // check acknowledgement flag
    if (!global_flag_pending)
        return;

    uint8_t status = global_acknowledgement_status;
    uint16_t seq = global_ack_sequence_number;

    send_frame(CODE_ACKNOWLEDGEMENT, seq, &status, 1);

    if (status == 0 && seq == global_sequence_number)	// correct data received
    {
        global_sequence_number++;
		if (!protocol_idle)			// Request next, if not idle
		{
			protocol_send_req();
		}

    }
    global_flag_pending = 0;
}
