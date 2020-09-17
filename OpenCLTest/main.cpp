#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/cl.hpp>
#include <fstream>
#include <iostream>
#include "..\OpenCLInitializer\Initializer.h"
using namespace std;


int main()
{
	int n, m, k;
	string inFile;
	/*cout << "Enter file name with data\n";
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


	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	auto platform = platforms.front();
	std::vector<cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

	auto device = devices.front();

	std::ifstream kernelFile("MatrixMultiplication.cl");
	std::string src(std::istreambuf_iterator<char>(kernelFile), (std::istreambuf_iterator<char>()));
	cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));

	cl::Context context(device);
	cl::Program program(context, sources);

	auto err = program.build("-cl-std=CL1.2");
	if (err == CL_SUCCESS)
		printf("\n Success build \n");


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

	cin.get();

	return 0;
}