#include "driver/ledc.h"
#include "esp_log.h"
#define CONFIG_CAMERA_MODEL_ESP_EYE 1
#include <cam.h>

static const char* TAG = "app_camera";

esp_err_t camera_init() {
#if CONFIG_CAMERA_MODEL_ESP_EYE || CONFIG_CAMERA_MODEL_ESP32_CAM_BOARD
    /* IO13, IO14 is designed for JTAG by default,
     * to use it as generalized input,
     * firstly declair it as pullup input */
    gpio_config_t conf;
    conf.mode = GPIO_MODE_INPUT;
    conf.pull_up_en = GPIO_PULLUP_ENABLE;
    conf.pull_down_en = GPIO_PULLDOWN_DISABLE;
    conf.intr_type = GPIO_INTR_DISABLE;
    conf.pin_bit_mask = 1LL << 13;
    gpio_config(&conf);
    conf.pin_bit_mask = 1LL << 14;
    gpio_config(&conf);
#endif

#ifdef CONFIG_LED_ILLUMINATOR_ENABLED
    gpio_set_direction(CONFIG_LED_LEDC_PIN, GPIO_MODE_OUTPUT);
    ledc_timer_config_t ledc_timer = {
        .duty_resolution = LEDC_TIMER_8_BIT, // resolution of PWM duty
        .freq_hz = 1000,                     // frequency of PWM signal
        .speed_mode = LEDC_LOW_SPEED_MODE,   // timer mode
        .timer_num = CONFIG_LED_LEDC_TIMER   // timer index
    };
    ledc_channel_config_t ledc_channel = {.channel = CONFIG_LED_LEDC_CHANNEL,
                                          .duty = 0,
                                          .gpio_num = CONFIG_LED_LEDC_PIN,
                                          .speed_mode = LEDC_LOW_SPEED_MODE,
                                          .hpoint = 0,
                                          .timer_sel = CONFIG_LED_LEDC_TIMER};
#ifdef CONFIG_LED_LEDC_HIGH_SPEED_MODE
    ledc_timer.speed_mode = ledc_channel.speed_mode = LEDC_HIGH_SPEED_MODE;
#endif
    switch (ledc_timer_config(&ledc_timer)) {
    case ESP_ERR_INVALID_ARG:
        ESP_LOGE(TAG, "ledc_timer_config() parameter error");
        break;
    case ESP_FAIL:
        ESP_LOGE(TAG, "ledc_timer_config() Can not find a proper pre-divider \
                       number base on the given frequency and the current duty_resolution");
        break;
    case ESP_OK:
        if (ledc_channel_config(&ledc_channel) == ESP_ERR_INVALID_ARG) {
            ESP_LOGE(TAG, "ledc_channel_config() parameter error");
        }
        break;
    default:
        break;
    }
#endif

    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sscb_sda = SIOD_GPIO_NUM;
    config.pin_sscb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;
    // init with high specs to pre-allocate larger buffers
    config.frame_size = FRAMESIZE_QSXGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;

    // camera init
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed with error 0x%x", err);
        return err;
    }

    sensor_t* s = esp_camera_sensor_get();
    s->set_vflip(s, 1); // flip it back
    // initial sensors are flipped vertically and colors are a bit saturated
    if (s->id.PID == OV3660_PID) {
        s->set_brightness(s, 1);  // up the blightness just a bit
        s->set_saturation(s, -2); // lower the saturation
    }
    // drop down frame size for higher initial frame rate
    s->set_framesize(s, FRAMESIZE_HD);

    return ESP_OK;
}

camera_fb_t* camera_capture() {
    // acquire a frame
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        ESP_LOGE(TAG, "Camera Capture Failed");
        return NULL;
    }

    return fb;
}

void camera_free(camera_fb_t* fb) {
    esp_camera_fb_return(fb);
}
