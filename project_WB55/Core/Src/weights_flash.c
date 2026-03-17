/*
 * weights_flash.c
 *
 *  Created on: Mar 17, 2026
 *      Author: yings
 */

#include "weights_flash.h"
#include "config.h"
#include "stm32wbxx_hal.h"
#include <string.h>
#include <stdint.h>

#if USE_BACKPROP
#include "nn.h"
#endif

#if USE_FF
#include "nn_ff.h"
#endif

#define WEIGHTS_MAGIC   0x59494E47UL   // YING
#define WEIGHTS_VERSION 0x00000005UL

#define NN_FLASH_SAVE_ADDR    0x08078000UL
#define NN_FLASH_SAVE_SIZE    (32U * 1024U)
#define NN_FLASH_SAVE_PAGES   (NN_FLASH_SAVE_SIZE / FLASH_PAGE_SIZE)

static float global_saved_infer_acc = 0.0f;

// [ magic ][ version ][ size ][ crc ][ acc ][ weights ]
typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint32_t version;
    uint32_t payload_size;
    uint32_t crc32;
} weights_header_t;


static uint8_t flash_rw_buffer[NN_FLASH_SAVE_SIZE];

static uint32_t crc32_simple(const uint8_t *data, uint32_t len)
{
    uint32_t crc = 0xFFFFFFFFUL;

    for (uint32_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (uint32_t j = 0; j < 8; j++) {
            if (crc & 1U) {
                crc = (crc >> 1) ^ 0xEDB88320UL;
            } else {
                crc >>= 1;
            }
        }
    }

    return ~crc;
}

static uint32_t flash_page_from_addr(uint32_t addr)
{
    return (addr - FLASH_BASE) / FLASH_PAGE_SIZE;
}

static uint32_t model_state_size(void)
{
#if USE_BACKPROP
    return (uint32_t)nn_state_size();
#elif USE_FF
    return (uint32_t)nn_ff_state_size();
#else
    return 0U;
#endif
}

static void model_state_export(void *dst)
{
#if USE_BACKPROP
    nn_state_export(dst);
#elif USE_FF
    nn_ff_state_export(dst);
#else
    (void)dst;
#endif
}

static uint8_t model_state_import(const void *src, uint32_t len)
{
#if USE_BACKPROP
    return nn_state_import(src, len);
#elif USE_FF
    return nn_ff_state_import(src, len);
#else
    (void)src;
    (void)len;
    return 0U;
#endif
}

uint8_t weights_flash_save(void)
{
    uint32_t model_size = model_state_size();
    uint32_t payload_size = sizeof(float) + model_size;

    if (model_size == 0U) {
        return 0U;
    }

    uint32_t total_size = sizeof(weights_header_t) + payload_size;
    uint32_t padded_size = (total_size + 7U) & ~7U;   // 8-byte align

    if (padded_size > NN_FLASH_SAVE_SIZE) {
        return 0U;
    }

    memset(flash_rw_buffer, 0xFF, padded_size);

    weights_header_t header;
    header.magic = WEIGHTS_MAGIC;
    header.version = WEIGHTS_VERSION;
    header.payload_size = payload_size;

    uint8_t *payload_ptr = flash_rw_buffer + sizeof(weights_header_t);

    memcpy(payload_ptr, &global_saved_infer_acc, sizeof(float));
    model_state_export(payload_ptr + sizeof(float));

    header.crc32 = crc32_simple(payload_ptr, payload_size);

    memcpy(flash_rw_buffer, &header, sizeof(header));

    HAL_FLASH_Unlock();

    FLASH_EraseInitTypeDef erase = {0};
    uint32_t page_error = 0;
    uint32_t needed_pages = (padded_size + FLASH_PAGE_SIZE - 1U) / FLASH_PAGE_SIZE;

    if (needed_pages > NN_FLASH_SAVE_PAGES) {
        HAL_FLASH_Lock();
        return 0U;
    }

    erase.TypeErase = FLASH_TYPEERASE_PAGES;
    erase.Page = flash_page_from_addr(NN_FLASH_SAVE_ADDR);
    erase.NbPages = needed_pages;
#ifdef FLASH_BANK_1
    erase.Banks = FLASH_BANK_1;
#endif

    if (HAL_FLASHEx_Erase(&erase, &page_error) != HAL_OK) {
        HAL_FLASH_Lock();
        return 0U;
    }

    for (uint32_t i = 0; i < padded_size; i += 8U) {
        uint64_t dw = 0xFFFFFFFFFFFFFFFFULL;
        memcpy(&dw, &flash_rw_buffer[i], sizeof(dw));

        if (HAL_FLASH_Program(FLASH_TYPEPROGRAM_DOUBLEWORD,
                              NN_FLASH_SAVE_ADDR + i,
                              dw) != HAL_OK) {
            HAL_FLASH_Lock();
            return 0U;
        }
    }

    HAL_FLASH_Lock();

    const weights_header_t *verify = (const weights_header_t *)NN_FLASH_SAVE_ADDR;

    if (verify->magic != WEIGHTS_MAGIC) {
        return 0U;
    }
    if (verify->version != WEIGHTS_VERSION) {
        return 0U;
    }
    if (verify->payload_size != payload_size) {
        return 0U;
    }
    if (verify->crc32 != header.crc32) {
        return 0U;
    }

    return 1U;
}

uint8_t weights_flash_load(void)
{
    const weights_header_t *header = (const weights_header_t *)NN_FLASH_SAVE_ADDR;

    if (header->magic != WEIGHTS_MAGIC) {
        return 0U;
    }
    if (header->version != WEIGHTS_VERSION) {
        return 0U;
    }

    uint32_t model_size = model_state_size();
    uint32_t expected = sizeof(float) + model_size;

    if (header->payload_size != expected) {
        return 0U;
    }

    if ((sizeof(weights_header_t) + header->payload_size) > NN_FLASH_SAVE_SIZE) {
        return 0U;
    }

    const uint8_t *payload = (const uint8_t *)(NN_FLASH_SAVE_ADDR + sizeof(weights_header_t));
    uint32_t crc = crc32_simple(payload, header->payload_size);

    if (crc != header->crc32) {
        return 0U;
    }

    memcpy(&global_saved_infer_acc, payload, sizeof(float));

    return model_state_import(payload + sizeof(float), model_size);
}

void weights_flash_set_infer_acc(float acc)
{
    global_saved_infer_acc = acc;
}

float weights_flash_get_infer_acc(void)
{
    return global_saved_infer_acc;
}
