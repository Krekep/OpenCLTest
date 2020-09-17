#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/cl.hpp>
#include <fstream>
#include <iostream>
#include "..\OpenCLInitializer\Initializer.h"
using namespace std;


int main()
{
	int n, m, k;
	/*string inFile;
	cout << "Enter file name with data\n";
	cin >> inFile;*/
	ifstream in("file.in");
	in >> m;
	in >> n;
	in >> k;

	bool* aMatr = (bool*)malloc(sizeof(bool) * m * k);
	bool* bMatr = (bool*)malloc(sizeof(bool) * k * n);
	bool* rMatr = (bool*)malloc(sizeof(bool) * m * n);


	for (int i = 0; i < m; i++)
		for (int j = 0; j < k; j++)
		{
			short temp;
			in >> temp;
			aMatr[i * k + j] = (temp == 1);
		}
	for (int i = 0; i < k; i++)
		for (int j = 0; j < n; j++)
		{
			short temp;
			in >> temp;
			bMatr[i * n + j] = (temp == 1);
		}
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			rMatr[i * n + j] = false;

	auto program = CreateProgram("MatrixMultiplication.cl");
	auto context = program.getInfo<CL_PROGRAM_CONTEXT>();
	auto devices = program.getInfo<CL_PROGRAM_DEVICES>();
	auto device = devices.front();


	cl::Buffer inputFirstBuf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(bool) * m * k, aMatr);
	cl::Buffer inputSecondBuf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(bool) * k * n, bMatr);
	cl::Buffer outputBuf(context, CL_MEM_READ_WRITE, sizeof(bool) * m * n);
	cl::Kernel kernel(program, "MatrixMultiplication");
	kernel.setArg(0, m);
	kernel.setArg(1, n);
	kernel.setArg(2, k);
	kernel.setArg(3, inputFirstBuf);
	kernel.setArg(4, inputSecondBuf);
	kernel.setArg(5, outputBuf);

	cl::CommandQueue queue(context, device);
	queue.enqueueNDRangeKernel(kernel, 0, cl::NDRange(m, n));

	queue.enqueueReadBuffer(outputBuf, CL_TRUE, 0, sizeof(bool) * m * n, rMatr, 0, NULL);
	cout << "Result of multiplication\n";
	for (int i = 0; i < m; i++, cout << endl)
		for (int j = 0; j < n; j++)
			if ((bool)rMatr[i * n + j])
				cout << "1 ";
			else
				cout << "0 ";
	cout << "--------------\n";

	bool continueCalc = true;
	bool* temp = (bool*)malloc(sizeof(bool) * m * k);
	bool* result = (bool*)malloc(sizeof(bool) * m * k);
	for (int i = 0; i < m; i++)
		for (int j = 0; j < k; j++)
		{
			temp[i * k + j] = aMatr[i * k + j];
			result[i * k + j] = aMatr[i * k + j];
		}

	while (continueCalc)
	{
		continueCalc = false;

		cl::Buffer inputTempBuf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(bool) * m * k, temp);
		kernel.setArg(0, m);
		kernel.setArg(1, m);
		kernel.setArg(2, m);
		kernel.setArg(3, inputFirstBuf);
		kernel.setArg(4, inputTempBuf);
		kernel.setArg(5, outputBuf);
		queue.enqueueNDRangeKernel(kernel, 0, cl::NDRange(m, n));
		queue.enqueueReadBuffer(outputBuf, CL_TRUE, 0, sizeof(bool) * m * n, temp, 0, NULL);

		for (int i = 0; i < m; i++)
			for (int j = 0; j < k; j++)
			{
				if (!result[i * k + j] && temp[i * k + j])
				{
					result[i * k + j] = true;
					continueCalc = true;
				}
			}
		
	}
	cout << "Result of transitive closure\n";
	for (int i = 0; i < m; i++, cout << endl)
		for (int j = 0; j < k; j++)
			cout << result[i * k + j] << " ";
	cout << "--------------\n";
	cin.get();
	return 0;
}