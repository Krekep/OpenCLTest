 // конвертер индексов матрицы в линейный адрес
#define IDX2LIN(i,j,l) (i+j*l)

__kernel void myGEMM1(const int M, const int N, const int K,
					const __global float* A, const __global float* B, __global float* C) 
{  // номер треда (2D решетка)
   const int r = get_global_id(0); // строка 0..M
   const int c = get_global_id(1); // столбец 0..N

   // вычисляем элемент [r,c] результирующей матрицы C
   float acc = 0.0f;
   for (int i=0; i<K; i++) {
	  acc += A[ IDX2LIN(r,i,M) ] * B[ IDX2LIN(i,c,K) ];
   }
   C[ IDX2LIN(r,c,M) ] = acc; // сохраняем результат
}