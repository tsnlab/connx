#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include "connx.h"
#include "hal.h"

static char* _strdup(char* str) {
    int len = strlen(str);
    char* str2 = connx_alloc(len + 1);
    if(str2 == NULL)
        return NULL;

    memcpy(str2, str, len + 1);

    return str2;
}

static char* next_token(char* text) {
    char* start = text;

    while(true) {
        if(*text == ' ' || *text == '\n') {
            *text = '\0';
            return start;
        } else if(*text == '\0') {
            text = start = text + 1;

            if(*start == '\0') // EOF
                return NULL;
        }

        text++;
    }
}

#define check_keyword(token, keyword)       \
    token = next_token(token);              \
    if(strcmp(token, keyword) != 0)         \
        return ILLEGAL_SYNTAX;

#define next_integer(token)                     \
    ({                                          \
        token = next_token(token);              \
        if(token == NULL)                       \
            return ILLEGAL_SYNTAX;              \
        strtol(token, NULL, 0);                 \
    })

#define next_string(token)                      \
    ({                                          \
        uint32_t len = next_integer(token);     \
        while(*token++ != '\0');                \
        token[len] = '\0';                      \
        token;                                  \
    })

static connx_ErrorCode parse_metadata(connx_Model* model, char* metadata) {
    char* token = metadata;

    // prase version
    check_keyword(token, "connx");

    model->version = next_integer(token);

    if(model->version <= 0 || model->version > 1)
        return NOT_SUPPORTED_CONNX_VERSION;

    // parse opset_import
    check_keyword(token, "opset_import");
    model->opset_count = next_integer(token);

    model->opset_names = connx_alloc(sizeof(char*) * model->opset_count);
    if(model->opset_names == NULL)
        return NOT_ENOUGH_MEMORY;

    model->opset_versions = connx_alloc(sizeof(uint32_t) * model->opset_count);
    if(model->opset_versions == NULL)
        return NOT_ENOUGH_MEMORY;

    // parse opset name, version
    for(uint32_t i = 0; i < model->opset_count; i++) {
        model->opset_names[i] = _strdup(next_string(token));
        if(model->opset_names[i] == NULL) {
            return NOT_ENOUGH_MEMORY;
        }
        model->opset_versions[i] = next_integer(token);
    }

    // parse graph
    check_keyword(token, "graph");
    model->graph_count = next_integer(token);

    // TODO: Parse graphs

    return OK;
}

int connx_Model_init(connx_Model* model) {
    // Parse metadata
    void* metadata = connx_load("model.connx");
    connx_ErrorCode ret = parse_metadata(model, metadata);
    connx_unload(metadata);

    if(ret != OK) {
        return ret;
    }

    return OK;
}

int connx_Model_destroy(connx_Model* model) {
    if(model->opset_versions != NULL) {
        connx_free(model->opset_versions);
    }

    if(model->opset_names != NULL) {
        for(uint32_t i = 0; i < model->opset_count; i++) {
            if(model->opset_names[i] != NULL) {
                connx_free(model->opset_names[i]);
            }
        }

        connx_free(model->opset_names);
    }

    return OK;
}

