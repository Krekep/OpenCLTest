kernel void MatrixMultiplication(const int M, const int N, const int K,
		 const global bool* A, const global bool* B, global bool* C) 
{  
   int iteratorM = get_global_id(0);
   int iteratorN = get_global_id(1);
   
   bool acc = false;
   for (int i = 0; i < K; i++) 
   {
	  acc = acc || (A[iteratorM + i * M] && B[i + iteratorN * K]);
   }
   C[iteratorM + iteratorN * M] = acc;
}



kernel void MatrixAddition(const int len,
		 const global bool* A, const global bool* B, global bool* C) 
{  
   int iteratorM = get_global_id(0);

   C[iteratorM] = (B[iteratorM] || A[iteratorM]);
}