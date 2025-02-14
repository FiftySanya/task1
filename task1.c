#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <omp.h>

#define MAX_NUMBERS 1000

typedef struct {
    int number;
    int frequency;
    int original_position;
} NumberInfo;

void calculate_frequencies(const int arr[], int size, NumberInfo result[]);
int compare_value_freq(const void *a, const void *b);
int compare_freq_value(const void *a, const void *b);
int partition(NumberInfo arr[], int low, int high, int (*compare_func)(const void *, const void *));
void parallel_qsort(NumberInfo arr[], int low, int high, int (*compare_func)(const void *, const void *));
void parallel_merge(NumberInfo arr[], int left, int mid, int right, int (*compare_func)(const void *, const void *));
void parallel_merge_sort(NumberInfo arr[], int left, int right, int (*compare_func)(const void *, const void *));
void heapify(NumberInfo arr[], int n, int i, int (*compare_func)(const void *, const void *));
void heap_sort(NumberInfo arr[], int n, int (*compare_func)(const void *, const void *));

int main(int argc, char *argv[]) {
    if (argc < 7) {
        printf("Insufficient number of arguments\n");
        printf("Usage: %s -t <sort_type> -k <sort_key> <numbers>\n", argv[0]);
        return 1;
    }
    
    int numbers[MAX_NUMBERS];
    int count = 0;
    char *sort_type = NULL;
    char *sort_key = NULL;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            sort_type = argv[++i];
        }
        else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
            sort_key = argv[++i];
        }
        else if (isdigit(argv[i][0]) || (argv[i][0] == '-' && isdigit(argv[i][1]))) {
            numbers[count++] = atoi(argv[i]);
        }        
    }
    
    if (!sort_type || !sort_key) {
        printf("Missing required parameters\n");
        return 1;
    }
    
    NumberInfo *number_info = (NumberInfo *)malloc(count * sizeof(NumberInfo));

    calculate_frequencies(numbers, count, number_info);
    
    omp_set_num_threads(omp_get_num_procs());
    
    int (*compare_func)(const void*, const void*) = strcmp(sort_key, "value-freq") == 0 ? compare_value_freq : compare_freq_value;

    
    if (strcmp(sort_type, "qsort") == 0) {
        parallel_qsort(number_info, 0, count - 1, compare_func);
    }
    else if (strcmp(sort_type, "merge") == 0) {
        parallel_merge_sort(number_info, 0, count - 1, compare_func);
    }
    else if (strcmp(sort_type, "heap") == 0) {
        heap_sort(number_info, count, compare_func);
    }
    else {
        printf("Unknown sort type: %s\n", sort_type);
        free(number_info);
        return 1;
    }

    
    printf("Sorted numbers (number:frequency): ");
    for (int i = 0; i < count; i++) {
        printf("%d:%d ", number_info[i].number, number_info[i].frequency);
    }
    printf("\n");
    
    free(number_info);
    return 0;
}

void calculate_frequencies(const int arr[], int size, NumberInfo result[]) {
    for (int i = 0; i < size; i++) {
        result[i].number = arr[i];
        result[i].frequency = 1;
        result[i].original_position = i;
        
        for (int j = 0; j < size; j++) {
            if (j != i && arr[j] == arr[i]) {
                result[i].frequency++;
            }
        }
    }
}

int compare_value_freq(const void *a, const void *b) {
    const NumberInfo *na = (const NumberInfo *)a;
    const NumberInfo *nb = (const NumberInfo *)b;
    
    if (na->number != nb->number)
        return na->number - nb->number;
    if (na->frequency != nb->frequency)
        return nb->frequency - na->frequency;
    return na->original_position - nb->original_position;
}

int compare_freq_value(const void *a, const void *b) {
    const NumberInfo *na = (const NumberInfo *)a;
    const NumberInfo *nb = (const NumberInfo *)b;
    
    if (na->frequency != nb->frequency)
        return nb->frequency - na->frequency;
    if (na->number != nb->number)
        return na->number - nb->number;
    return na->original_position - nb->original_position;
}

int partition(NumberInfo arr[], int low, int high, int (*compare_func)(const void *, const void *)) {
    NumberInfo pivot = arr[high];
    int i = low - 1;
    
    for (int j = low; j < high; j++) {
        if (compare_func(&arr[j], &pivot) <= 0) {
            i++;
            NumberInfo temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }
    
    NumberInfo temp = arr[i + 1];
    arr[i + 1] = arr[high];
    arr[high] = temp;
    
    return i + 1;
}

void parallel_qsort(NumberInfo arr[], int low, int high, int (*compare_func)(const void *, const void *)) {
    if (low < high) {
        int pivot = partition(arr, low, high, compare_func);
        
        #pragma omp parallel sections
        {
            #pragma omp section
            parallel_qsort(arr, low, pivot - 1, compare_func);
            #pragma omp section
            parallel_qsort(arr, pivot + 1, high, compare_func);
        }
    }
}

void parallel_merge(NumberInfo arr[], int left, int mid, int right, int (*compare_func)(const void *, const void *)) {
    int n1 = mid - left + 1;
    int n2 = right - mid;
    
    NumberInfo *leftArr = (NumberInfo *)malloc(n1 * sizeof(NumberInfo));
    NumberInfo *rightArr = (NumberInfo *)malloc(n2 * sizeof(NumberInfo));
    
    #pragma omp parallel sections
    {
        #pragma omp section
        memcpy(leftArr, &arr[left], n1 * sizeof(NumberInfo));
        #pragma omp section
        memcpy(rightArr, &arr[mid + 1], n2 * sizeof(NumberInfo));
    }
    
    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (compare_func(&leftArr[i], &rightArr[j]) <= 0)
            arr[k++] = leftArr[i++];
        else
            arr[k++] = rightArr[j++];
    }
    
    while (i < n1) arr[k++] = leftArr[i++];
    while (j < n2) arr[k++] = rightArr[j++];
    
    free(leftArr);
    free(rightArr);
}

void parallel_merge_sort(NumberInfo arr[], int left, int right, int (*compare_func)(const void *, const void *)) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        
        #pragma omp parallel sections
        {
            #pragma omp section
            parallel_merge_sort(arr, left, mid, compare_func);
            #pragma omp section
            parallel_merge_sort(arr, mid + 1, right, compare_func);
        }
        
        parallel_merge(arr, left, mid, right, compare_func);
    }
}

void heapify(NumberInfo arr[], int n, int i, int (*compare_func)(const void *, const void *)) {
    int current = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;
    
    if (left < n && compare_func(&arr[left], &arr[current]) > 0)
        current = left;
        
    if (right < n && compare_func(&arr[right], &arr[current]) > 0)
        current = right;
    
    if (current != i) {
        NumberInfo temp = arr[i];
        arr[i] = arr[current];
        arr[current] = temp;
        heapify(arr, n, current, compare_func);
    }
}

void heap_sort(NumberInfo arr[], int n, int (*compare_func)(const void *, const void *)) {
    for (int i = n / 2 - 1; i >= 0; i--) {
        heapify(arr, n, i, compare_func);
    }
 
    for (int i = n - 1; i >= 0; i--) {
        NumberInfo temp = arr[0];
        arr[0] = arr[i];
        arr[i] = temp;
 
        if (i > 0) {
            heapify(arr, i, 0, compare_func);
        }
    }
}
