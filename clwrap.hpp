#ifndef __CLWRAP_HPP_DEFINED__
#define __CLWRAP_HPP_DEFINED__

// clwrap.hpp is a C++ header file that wraps OpenCL host boilerplate
// codes and provides some performance and power measurement hooks.
// 
// Written by Kaz Yoshii <ky@anl.gov>
//
// Tested platform:
// Intel OpenCL SDK for embedded GPUs
// Intel OpenCL SDK for FPGAs (e.g., Nallatech 385A)
//

#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <CL/cl.hpp>

// e.g., local referece
// /soft/fpga/altera/pro/16.0.2.222/hld/host/include/CL/cl.hpp


// uncomment below for reading the board power. only tested this on Nallatech385A

#define ENABLE_AOCL_MMD_HACK
#ifdef ENABLE_AOCL_MMD_HACK
extern "C" void aocl_mmd_card_info(const char *name , int id, 
					   size_t sz, 
					   void *v, size_t *retsize);
#endif

using namespace std;

class clWrap {
public:
	// VALUE: pass by value, otherwise passed by reference
	enum dir_enum { VALUE, HOST2DEV, DEV2HOST, DUPLEX };
	struct arg_struct {
		dir_enum dir;
		size_t sz;
		void *data;
		bool buffered;
		cl::Buffer buf;
	};

private:
	vector<cl::Platform> pfs;
	vector<cl::Device> devs;
	vector<cl::Program> prgs;
	cl::Context ctx;
	int platform_id, device_id, program_id;

	cl::Event kernel_event;
	cl::CommandQueue queue;
	cl::Kernel kernel;
	vector<struct arg_struct> kargs;

public:
	char *loadfile(string fn, cl_ulong &sz)
	{
		struct stat st;

		sz = 0;

		stat(fn.c_str(), &st);

		if (!S_ISREG(st.st_mode)) {
			cout << fn << " is not a regular file!" << endl;
			return NULL;
		}

		// load binary and build prgs
		ifstream f0(fn.c_str(), ifstream::binary);

		if (! f0.good()) {
			cout << "Unable to load " << fn << endl;
			return NULL;
		}
		f0.seekg(0, f0.end);
		sz = f0.tellg();

		f0.seekg(0, f0.beg);
	
		char *f0c = new char [sz+1];
		f0.read(f0c, sz);
		f0c[sz] = 0;
		return f0c;
	}

#ifdef ENABLE_INTELFPGA
	bool loadprog(string fn) {
		cl_ulong sz;
		char *binfile = loadfile(fn, sz);
		if (! binfile)
			return false;
		cl::Program::Binaries bin;
		bin.push_back({binfile,sz});
		prgs.push_back(cl::Program(ctx,devs,bin));
		return true;
	}
#elif ENABLE_INTELGPU
	bool loadprog(string fn) {
		cl_ulong sz;
		char *srcfile = loadfile(fn, sz);
		cl_int err = CL_SUCCESS;

		if (! srcfile)
			return false;
		cl::Program::Sources src;
		src.push_back({srcfile,sz});
		cl::Program p(ctx, src, &err);
		if (err != CL_SUCCESS) {
			cout << "Program failed" << err << endl;
			return false;
		}
		p.build(devs);
		prgs.push_back(p);

		return true;
        }
#else
#error "Add -DENABLE_INTELFPGA or -DENABLE_INTELGPU to compiler options"
#endif

#ifdef AOCL_MMD_HACK
	// technically this function should be called in other thread context
	// to measure the power consumption while kernel is running.
	float readboardpower(void) {
		float pwr;
		size_t retsize;

		aocl_mmd_card_info("aclnalla_pcie0", 9,
				   sizeof(float),
				   (void*)&pwr, &retsize);

		return pwr;
	}
#else
	float readboardpower(void) {
		return 0.0;
	}
#endif

	clWrap(int pid=0, int did=0) {
		cl::Platform::get(&pfs);
		if (pfs.size() == 0) {
			cout << "No platform found" << endl;
			return;
		}

		/* set default platform id and device id */
		platform_id = pid;

#ifdef ENABLE_INTELGPU
		for (int i = 0; i < (int)pfs.size(); i++) {
			string pn = pfs[i].getInfo<CL_PLATFORM_NAME>();
			if (pn.find("Intel Gen OCL") != string::npos) {
				platform_id = i;
				break;
			}
		}
#endif

		pfs[platform_id].getDevices(CL_DEVICE_TYPE_ALL, &devs);
		if (devs.size() == 0) {
			cout << "No device found" << endl;
			return;
		}

		ctx = devs;

		device_id = did;
		program_id = 0; //
	}

	void listPlatforms(void) {
		cout << "[Platforms]\n";
		for (int i = 0; i < (int)pfs.size(); i++) {
			cout << i << ": " << pfs[i].getInfo<CL_PLATFORM_NAME>();
			if (i == platform_id) cout << " [selected]";
			cout << endl;
		}
	}

	void listDevices(void) {
		cout << "[Devices]\n";
		for (int i = 0; i < (int)devs.size(); i++) {
			cout << "Device" << i << ": " << devs[i].getInfo<CL_DEVICE_NAME>();
			// cout << " " << devs[i].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << " ";
			if (i == device_id) cout << " [selected]";
			cout << endl;
		}
	}

	bool prepKernel(const char *filename, const char *funcname) {
		cl_int err = CL_SUCCESS;

		string fn = filename;
		size_t pos = fn.find_last_of(".");

		if (pos == std::string::npos) {
#ifdef ENABLE_INTELFPGA
			fn = fn + ".aocx";
#elif  ENABLE_INTELGPU
			fn = fn + ".cl";
#endif
		}

		if (! loadprog(fn)) {
			return false;
		}

		queue = cl::CommandQueue(ctx, devs[device_id], 0, &err);
		kernel = cl::Kernel(prgs[program_id], funcname, &err);
		if (err != CL_SUCCESS) {
			switch(err) {
			case CL_INVALID_PROGRAM: cout << "CL_INVALID_PROGRAM\n"; break;
			case CL_INVALID_PROGRAM_EXECUTABLE: cout << "CL_INVALID_PROGRAM_EXECUTABLE\n"; break;
			case CL_INVALID_KERNEL_NAME: cout << "CL_INVALID_KERNEL_NAME\n"; break;
			case CL_INVALID_KERNEL_DEFINITION: cout << "CL_INVALID_KERNEL_DEFINITION\n"; break;
			default:
				cout << "cl::Kernel() failed:" << err << endl;
			}
			return false;
		}
		return true;
	}

	/* */
	cl::Kernel getKernel(void) { return kernel; }
	cl::CommandQueue  getQueue(void)  { return queue; }
	cl::Context  getContext(void)  { return ctx; }


	cl_mem_flags  get_mem_flag(dir_enum  dir) {
		cl_mem_flags rc = CL_MEM_READ_WRITE;

		switch (dir) {
		case HOST2DEV:
			rc = CL_MEM_READ_ONLY; break;
		case DEV2HOST:
			rc = CL_MEM_WRITE_ONLY; break;
		default:
			rc = CL_MEM_READ_WRITE;
		}

		return rc;
	}

	// return the index of the added argument
	int appendArg(size_t sz, void *data,
		      dir_enum  dir = VALUE) {


		bool buffered = false;

		if (dir != VALUE) buffered = true;

		kargs.push_back(arg_struct());

		int idx = kargs.size() - 1;
		kargs[idx].sz = sz;
		kargs[idx].data = data;
		kargs[idx].dir = dir;
		kargs[idx].buffered = buffered;

		if (buffered) {
			cl::Buffer buf(ctx, get_mem_flag(dir), sz);
			kargs[idx].buf = buf;
			kernel.setArg(idx, buf);
		} else {
			kernel.setArg(idx, sz, data);
		}

		return idx;
	}



	void runKernel(cl::NDRange &gsz, cl::NDRange &lsz) {
		// copy data to dev
		// do timing later
		for (vector<arg_struct>::iterator it = kargs.begin(); it != kargs.end(); ++it) {
			if (it->dir == HOST2DEV || it->dir == DUPLEX)  {
				queue.enqueueWriteBuffer(it->buf, CL_TRUE, 0, it->sz, it->data);
			}
		}

		queue.enqueueNDRangeKernel(
				       kernel,
				       cl::NullRange, // offset
				       gsz,
				       lsz,
				       NULL, // events
				       &kernel_event);
		kernel_event.wait();

		// copy data from dev
		// do timing later
		for (vector<arg_struct>::iterator it = kargs.begin(); it != kargs.end(); ++it) {
			if (it->dir == DEV2HOST || it->dir == DUPLEX)  {
				queue.enqueueReadBuffer(it->buf, CL_TRUE, 0, it->sz, it->data);
			}
		}
	}

	void runKernel(int gsz, int lsz = 1) {
		cl::NDRange ngsz(gsz);
		cl::NDRange nlsz(lsz);
		runKernel(ngsz, nlsz);
	}

	double getKernelElapsedNanoSec(void) {
		cl_ulong start, end;
		kernel_event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
		kernel_event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
		return end - start;
	}

};

#endif
