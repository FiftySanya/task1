# task1
Програма сортування списку цілих чисел за зростанням різними способами на мові C
# Паралельна Реалізація Сортування

Ця програма на C реалізує три різні паралельні алгоритми сортування (QuickSort, MergeSort та HeapSort) з можливістю сортування чисел на основі їх значень та частоти появи. Програма використовує OpenMP для паралельного виконання, щоб покращити продуктивність на багатоядерних системах.

## Особливості

- Три алгоритми сортування: QuickSort, MergeSort та HeapSort
- Два критерії сортування: значення-частота та частота-значення
- Паралельне виконання за допомогою OpenMP
- Збереження початкових позицій елементів для стабільного сортування
- Підтримка як додатних, так і від'ємних цілих чисел

## Використання

```bash
./program -t <тип_сортування> -k <ключ_сортування> <числа>
```

### Параметри

- `-t`: Тип алгоритму сортування (qsort, merge або heap)
- `-k`: Ключ сортування (value-freq або freq-value)
- `<числа>`: Список цілих чисел, розділених пробілами

### Приклад

```bash
./program -t merge -k value-freq 1 2 2 3 3 3 4 4 4 4
```

## Структури Даних

### Структура NumberInfo

```c
typedef struct {
    int number;           // Саме число
    int frequency;        // Скільки разів число зустрічається
    int original_position; // Початкова позиція в масиві
} NumberInfo;
```

## Опис Функцій

### Основні Функції Обробки

#### `calculate_frequencies`
```c
void calculate_frequencies(const int arr[], int size, NumberInfo result[]);
```
- Обчислює частоту кожного числа у вхідному масиві
- Зберігає результати в масиві структур NumberInfo
- Часова складність: O(n²)

#### Функції Порівняння

##### `compare_value_freq`
```c
int compare_value_freq(const void *a, const void *b);
```
- Порівнює числа в першу чергу за значенням
- Якщо значення рівні, порівнює за частотою (вища частота першою)
- Якщо частоти рівні, зберігає початковий порядок

##### `compare_freq_value`
```c
int compare_freq_value(const void *a, const void *b);
```
- Порівнює числа в першу чергу за частотою (вища частота першою)
- Якщо частоти рівні, порівнює за значенням
- Якщо значення рівні, зберігає початковий порядок

### Реалізації Паралельного Сортування

#### Реалізація QuickSort

##### `parallel_qsort`
```c
void parallel_qsort(NumberInfo arr[], int low, int high, int (*compare_func)(const void *, const void *));
```
- Паралельна реалізація QuickSort
- Використовує секції OpenMP для паралельних рекурсивних викликів
- Паралелізація відбувається на межах розділення

##### `partition`
```c
int partition(NumberInfo arr[], int low, int high, int (*compare_func)(const void *, const void *));
```
- Реалізує логіку розділення для QuickSort
- Вибирає останній елемент як опорний
- Розташовує елементи навколо опорного елемента

#### Реалізація MergeSort

##### `parallel_merge_sort`
```c
void parallel_merge_sort(NumberInfo arr[], int left, int right, int (*compare_func)(const void *, const void *));
```
- Паралельна реалізація MergeSort
- Використовує секції OpenMP для паралельних рекурсивних викликів
- Паралелізує як фази поділу, так і злиття

##### `parallel_merge`
```c
void parallel_merge(NumberInfo arr[], int left, int mid, int right, int (*compare_func)(const void *, const void *));
```
- Реалізує паралельне злиття двох відсортованих підмасивів
- Використовує паралельні секції для копіювання даних у тимчасові масиви
- Виконує фактичне злиття послідовно для стабільності

#### Реалізація HeapSort

##### `parallel_heap_sort`
```c
void parallel_heap_sort(NumberInfo arr[], int n, int (*compare_func)(const void *, const void *));
```
- Реалізує алгоритм HeapSort
- Паралелізація обмежена через послідовну природу операцій з купою
- Використовує heapify для підтримки властивостей купи

##### `heapify`
```c
void heapify(NumberInfo arr[], int n, int i, int (*compare_func)(const void *, const void *));
```
- Підтримує властивості купи для алгоритму сортування
- Рекурсивно забезпечує розміщення найбільшого елемента у корені
- Використовує надану функцію порівняння для впорядкування елементів

## Деталі Паралелізації

Програма реалізує паралелізацію за допомогою OpenMP кількома способами:

1. **Керування Потоками**
   - Використовує `omp_set_num_threads(omp_get_num_procs())` для встановлення кількості потоків рівній кількості доступних процесорів
   - Автоматично керує створенням та знищенням потоків

2. **Паралельні Секції**
   - QuickSort: Паралельна обробка підмасивів після розділення
   - MergeSort: Паралельна обробка лівої та правої половин
   - Паралельне копіювання даних в операціях злиття

3. **Синхронізація**
   - Неявна синхронізація в кінці паралельних секцій
   - Природні точки синхронізації у фазі злиття MergeSort

## Міркування щодо Продуктивності

- QuickSort зазвичай працює найкраще для випадкових даних
- MergeSort забезпечує стабільне сортування, але потребує додаткової пам'яті
- HeapSort має обмежену паралелізацію, але гарантовану складність O(n log n)
- Ефективність паралелізації залежить від:
  - Розміру вхідних даних
  - Кількості доступних процесорів
  - Розподілу даних

## Використання Пам'яті

- MergeSort потребує O(n) додаткового простору
- QuickSort потребує O(log n) стекового простору
- HeapSort потребує O(1) додаткового простору
- Додатковий простір O(n) для масиву структур NumberInfo

## Обробка Помилок

- Перевірка достатньої кількості аргументів командного рядка
- Перевірка типу сортування та параметрів ключа
- Правильне виділення та звільнення пам'яті
- Максимальний розмір вхідних даних обмежений MAX_NUMBERS (1000)
