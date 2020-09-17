#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL\cl.hpp>
#include <fstream>
#include <iostream>

cl::Program CreateProgram(const std::string& file)
{
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	auto platform = platforms.front();
	std::vector<cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

	auto device = devices.front();

	std::ifstream kernelFile(file);
	std::string src(std::istreambuf_iterator<char>(kernelFile), (std::istreambuf_iterator<char>()));
	cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));

	cl::Context context(device);
	cl::Program program(context, sources);

	auto err = program.build("-cl-std=CL1.2");
	if (err == CL_SUCCESS) 
		printf("\n Success build \n");
	return program;
}

