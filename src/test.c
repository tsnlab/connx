#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include "tensor.h"

static void test_tensor_alloc(__attribute__((unused)) void** state) {
    int32_t shape[] = { 1, 2, 3 };
    connx_Tensor* tensor = connx_Tensor_alloc(FLOAT32, 3, shape);

    assert_true(tensor->ndim == 3);

    for(int i = 0; i < 3; i++)
        assert_true(tensor->shape[i] == i + 1);

    connx_Tensor_unref(tensor);
}

static void test_iterator(__attribute__((unused)) void** state) {
    int32_t shape[] = { 1, 2, 3 };
    connx_Tensor* tensor = connx_Tensor_alloc(FLOAT32, 3, shape);

    int32_t iter_size = connx_Iterator_size(tensor);
    assert_true(iter_size == 1 + 4 * 3);

    connx_Tensor_unref(tensor);

    int32_t iterator[iter_size];
    int32_t start[] = { 0, 0, 0 };
    int32_t stop[] = { 1, 2, 3 };
    int32_t step[] = { 1, 1, 1 };

    connx_Iterator_init(iterator, 3, start, stop, step);
    assert_true(connx_Iterator_ndim(iterator) == 3);

    for(int i = 0; i < 3; i++)
        assert_true(connx_Iterator_start(iterator)[i] == start[i]);

    for(int i = 0; i < 3; i++)
        assert_true(connx_Iterator_stop(iterator)[i] == stop[i]);

    for(int i = 0; i < 3; i++)
        assert_true(connx_Iterator_step(iterator)[i] == step[i]);

    for(int i = 0; i < 3; i++)
        assert_true(connx_Iterator_index(iterator)[i] == (i == 2 ? -1 : 0));

    assert_true(connx_Iterator_next(iterator));
    assert_true(connx_Iterator_index(iterator)[0] == 0);
    assert_true(connx_Iterator_index(iterator)[1] == 0);
    assert_true(connx_Iterator_index(iterator)[2] == 0);

    assert_true(connx_Iterator_next(iterator));
    assert_true(connx_Iterator_index(iterator)[0] == 0);
    assert_true(connx_Iterator_index(iterator)[1] == 0);
    assert_true(connx_Iterator_index(iterator)[2] == 1);

    assert_true(connx_Iterator_next(iterator));
    assert_true(connx_Iterator_index(iterator)[0] == 0);
    assert_true(connx_Iterator_index(iterator)[1] == 0);
    assert_true(connx_Iterator_index(iterator)[2] == 2);

    assert_true(connx_Iterator_next(iterator));
    assert_true(connx_Iterator_index(iterator)[0] == 0);
    assert_true(connx_Iterator_index(iterator)[1] == 1);
    assert_true(connx_Iterator_index(iterator)[2] == 0);

    assert_true(connx_Iterator_next(iterator));
    assert_true(connx_Iterator_index(iterator)[0] == 0);
    assert_true(connx_Iterator_index(iterator)[1] == 1);
    assert_true(connx_Iterator_index(iterator)[2] == 1);

    assert_true(connx_Iterator_next(iterator));
    assert_true(connx_Iterator_index(iterator)[0] == 0);
    assert_true(connx_Iterator_index(iterator)[1] == 1);
    assert_true(connx_Iterator_index(iterator)[2] == 2);

    assert_true(!connx_Iterator_next(iterator));
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_tensor_alloc),
        cmocka_unit_test(test_iterator),
    };

    return cmocka_run_group_tests(tests, NULL, NULL);
}
