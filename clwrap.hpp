#ifndef __CLWRAP_HPP_DEFINED__
#define __CLWRAP_HPP_DEFINED__

// clwrap.hpp is a C++ header file that wraps OpenCL host boilerplate
// codes and provides some performance and power measurement hooks.
//
// Written by Kaz Yoshii <kazutomo@mcs.anl.gov>
//
// Tested platforms:
// Intel OpenCL SDK for Intel embedded GPUs (Gen9)
// Intel OpenCL SDK for FPGAs (e.g., Nallatech 385A)
//
// LICENSE: BSD 3-clause
//

#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
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

class clwrap {
public:
	const int version_major = 0;
	const int version_minor = 5;

	// VALUE: pass by value, otherwise passed by reference
	enum dir_enum { VALUE, HOST2DEV, DEV2HOST, DUPLEX };
	struct arg_struct {
		dir_enum dir;
		size_t sz;
		void *data;
		bool buffered;
		cl::Buffer buf;
	};

	typedef const char* here_t;
private:
	std::vector<cl::Platform> pfs; // initialized only in c'tor
	std::vector<cl::Device> devs; // initialized only in c'tor
	std::vector<cl::Device> dev_selected; // initialized only in c'tor
	std::vector<cl::Program> prgs; // clear and push_back in prepKernel()
	cl::Context ctx;
	// selected ids
	int platform_id, device_id, program_id;

	cl::Event kernel_event;
	cl::CommandQueue queue;
	cl::Kernel kernel;
	std::vector<struct arg_struct> kargs;
	std::vector<char>  kernelbuf;

	enum loader {
		     SRC=0,
		     BIN=1
	};
	typedef std::pair<std::string,enum loader> kext_t;
	std::vector<kext_t>  kexts;

	bool flag_dumpkernel;
	int flag_verbose;

	const char* getkernelbuf()
	{
		return &kernelbuf[0];
	}
	size_t  getkernelbufsize()
	{
		return kernelbuf.size();
	}

	// return 0 if success
	bool loadkernel(std::string fn, bool nullterminate=false)
	{
		size_t sz;

		std::ifstream f(fn.c_str(), std::ifstream::binary);

		if (! f.is_open()) {
			return false;
		}

		f.seekg(0, f.end);
		sz = f.tellg();
		f.seekg(0, f.beg);

		if (nullterminate)
			kernelbuf.resize(sz+1);
		else
			kernelbuf.resize(sz);

		f.read(&kernelbuf[0], sz);

		if (nullterminate)
			kernelbuf[sz] = 0;

		f.close();
		return true;
	}

	bool dumpkernel(std::string fn, const cl_ulong sz, const char *buf)
	{
		std::ofstream f(fn.c_str(), std::ostream::binary);
		f.write(buf, sz);
		return true;
	}

	bool fexists(const std::string fn)
	{
		struct stat st;
		return (stat(fn.c_str(), &st) == 0);
	}

	bool loadprog_bin(std::string fn) {
		if (! loadkernel(fn))
			return false;

		cl::Program::Binaries bin;
		bin.push_back({getkernelbuf(),getkernelbufsize()});

		std::vector<int> binaryStatus;
		cl_int err = CL_SUCCESS;

		cl::Program p(ctx, dev_selected, bin, &binaryStatus, &err);
		// std::cout << "err=" << err << std::endl;

		if (err != CL_SUCCESS) {
			std::cout << "fn=" << fn << std::endl;
			std::cout << "Program failed to build: " << err << std::endl;
			std::cout << p.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev_selected[0]);
			return false;
		}
		prgs.push_back(p);

		if (flag_verbose > 0) {
			std::cout << "binary prog: " << fn << " is loaded.\n";
		}

		return true;
	}

	bool loadprog_src(std::string fn) {
		cl_int err = CL_SUCCESS;

		if (! loadkernel(fn, true))
			return false;

		cl::Program::Sources src;
		src.push_back({getkernelbuf(),getkernelbufsize()});
		cl::Program p(ctx, src, &err);
		if (err != CL_SUCCESS) {
			std::cout << "Program failed" << err << std::endl;
			return false;
		}
		err = p.build(dev_selected);
		// err = p.build(dev_selected, "-cl-intel-gtpin-rera");
		if (err != CL_SUCCESS) {
			std::cout << "Program failed to build: " << err << std::endl;
			std::cout << p.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev_selected[0]);
			return false;
		}
		prgs.push_back(p);

		if (flag_verbose > 0) {
			std::cout << "text prog: " << fn << " is loaded.\n";
		}
#if 0
		std::vector<unsigned long> kszs =
			p.getInfo<CL_PROGRAM_BINARY_SIZES>();
		std::vector<char*> bins;
		bins.push_back( new char[*kszs.begin()] );
		p.getInfo(CL_PROGRAM_BINARIES, &bins[0]);
		savefile("test.bin", *kszs.begin(), bins[0] );
#endif
		return true;
        }

public:

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
	// constructor
	clwrap(int pid=0, int did=0) {

		kexts = {{".aocx", BIN},
			 {".bin", BIN},
			 {".cl", SRC}
		};

		flag_dumpkernel = false;
		flag_verbose = 0;

		if(const char *env = std::getenv("CLW_VERBOSE"))
			flag_verbose = std::atoi(env);
		if(std::getenv("CLW_DUMPKERNEL"))
			flag_verbose = true;

		cl::Platform::get(&pfs);
		if (pfs.size() == 0) {
			std::cout << "No platform found" << std::endl;
			return;
		}

		/* set default platform id and device id */
		platform_id = pid;


		/*
		  platform id

		  Intel CPU: "Intel(R) CPU Runtime for OpenCL(TM) Applications"
		  Intel CPU: "Experimental OpenCL 2.1 CPU Only Platform"
		  Intel Gen GPU (NEO): "Intel(R) OpenCL HD Graphics"
		  Intel FPGA: "Intel(R) FPGA SDK for OpenCL(TM)"
		  pocl: "Portable Computing Language"
		 */
		std::string pfkey = "Intel";
		if(const char *env = std::getenv("CLW_PF")) {
			pfkey = env;
			std::cout << "Platform search: " << pfkey << std::endl;
		}
		platform_id = -1;
		for (int i = 0; i < (int)pfs.size(); i++) {
			std::string pn = pfs[i].getInfo<CL_PLATFORM_NAME>();
			if (pn.find(pfkey.c_str()) != std::string::npos) {
				platform_id = i;
				break;
			}
		}
		if (platform_id < 0) {
			std::cout << "No platform found" << std::endl;
			return;
		}

		pfs[platform_id].getDevices(CL_DEVICE_TYPE_ALL, &devs);
		if (devs.size() == 0) {
			std::cout << "No device found" << std::endl;
			return;
		}

		ctx = devs;
		device_id = did;

		dev_selected.push_back(devs[device_id]);

		program_id = 0; //

	}

	void listPlatforms(void) {
		std::cout << "[Platforms]\n";
		for (int i = 0; i < (int)pfs.size(); i++) {
			std::cout << i << ": " << pfs[i].getInfo<CL_PLATFORM_NAME>();
			if (i == platform_id) std::cout << " [selected]";
			std::cout << std::endl;
		}
	}

	void listDevices(void) {
		std::cout << "[Devices]\n";
		for (int i = 0; i < (int)devs.size(); i++) {
			std::cout << "Device" << i << ": " << devs[i].getInfo<CL_DEVICE_NAME>();
			// std::cout << " " << devs[i].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << " ";
			if (i == device_id) std::cout << " [selected]";
#if 0
			// to query shared virtual memory capabilities
			cl_device_svm_capabilities svmcap;
			devs[i].getInfo(CL_DEVICE_SVM_CAPABILITIES, &svmcap);
			// CL_INVALID_VALUE is returned, no svm is supported
			if( svmcap&CL_DEVICE_SVM_COARSE_GRAIN_BUFFER ) std::cout << "CGBUF ";
			if( svmcap&CL_DEVICE_SVM_FINE_GRAIN_BUFFER ) std::cout << "FGBUF ";
			if( svmcap&CL_DEVICE_SVM_FINE_GRAIN_SYSTEM ) std::cout << "FGSYS ";
			if( svmcap&CL_DEVICE_SVM_ATOMICS ) std::cout << "ATOM "; // only for fine grain
#endif
			std::cout << std::endl;
		}
	}
	void info(void) {
		std::cout << "clwrap version " << version_major << "." << version_minor << std::endl;
		listPlatforms();
		listDevices();
	}

	bool prepKernel(const char *filename, const char *funcname = NULL) {
		std::string fn = filename;
		cl_int err = CL_SUCCESS;
		size_t pos = fn.find_last_of(".");
		kext_t kext_found = {"", BIN};

		prgs.clear();

		if (pos == std::string::npos) {
			// the file extension is omitted
			std::string tmpfn;

			if (!funcname)  funcname = filename;

			for (std::vector<kext_t>::iterator it = kexts.begin(); it != kexts.end(); ++it)  {
				tmpfn = fn + (*it).first;
				if (fexists(tmpfn)) {
					fn = tmpfn;
					kext_found = *it;
					break;
				}
			}
		} else {
			// assume that the file extension is explicitly specified
			std::string e = fn.substr(pos);

			for (std::vector<kext_t>::iterator it = kexts.begin(); it != kexts.end(); ++it)  {
				if (e == (*it).first) {
					kext_found = *it;
					break;
				}
			}

		}
		if (kext_found.first == "") {
			std::cout << "Error: no kernel file found! " << filename << std::endl;
			return false;
		}

		if (kext_found.second == BIN) {
			if (! loadprog_bin(fn)) {
				return false;
			}
		} else {
			if (! loadprog_src(fn)) {
				return false;
			}
		}

		// create a command queue and kernel

		queue = cl::CommandQueue(ctx, dev_selected[0], CL_QUEUE_PROFILING_ENABLE, &err);
		kernel = cl::Kernel(prgs[program_id], funcname, &err);
		if (err != CL_SUCCESS) {
			switch(err) {
			case CL_INVALID_PROGRAM: std::cout << "CL_INVALID_PROGRAM\n"; break;
			case CL_INVALID_PROGRAM_EXECUTABLE: std::cout << "CL_INVALID_PROGRAM_EXECUTABLE\n"; break;
			case CL_INVALID_KERNEL_NAME: std::cout << "CL_INVALID_KERNEL_NAME\n"; break;
			case CL_INVALID_KERNEL_DEFINITION: std::cout << "CL_INVALID_KERNEL_DEFINITION\n"; break;
			default:
				std::cout << "cl::Kernel() failed:" << err << std::endl;
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

	void write_to_dev(void) {
		for (std::vector<arg_struct>::iterator it = kargs.begin(); it != kargs.end(); ++it) {
			if (it->dir == HOST2DEV || it->dir == DUPLEX)  {
				queue.enqueueWriteBuffer(it->buf, CL_TRUE, 0, it->sz, it->data);
			}
		}
	}

	void read_from_dev(void) {
		for (std::vector<arg_struct>::iterator it = kargs.begin(); it != kargs.end(); ++it) {
			if (it->dir == DEV2HOST || it->dir == DUPLEX)  {
				queue.enqueueReadBuffer(it->buf, CL_TRUE, 0, it->sz, it->data);
			}
		}
	}

	void runProducer(void) {

		cl::NDRange gsz(1);
		cl::NDRange lsz(1);

		write_to_dev();

		queue.enqueueNDRangeKernel(
				       kernel,
				       cl::NullRange, // offset
				       gsz,
				       lsz,
				       NULL,
				       NULL);
	}

	void runKernel(cl::NDRange &gsz, cl::NDRange &lsz) {
		write_to_dev();

		queue.enqueueNDRangeKernel(
				       kernel,
				       cl::NullRange, // offset
				       gsz,
				       lsz,
				       NULL, // events
				       &kernel_event);
		kernel_event.wait();

		read_from_dev();

		queue.finish(); // needed?
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
