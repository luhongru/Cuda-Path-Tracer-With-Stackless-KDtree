#include "profile.h"

Profiler::Profiler() {}

Profiler::~Profiler() {
	*fs << partition_time / count << " " << kernel_time / count << " " << time / count << "\n";
	fs->close();
	delete fs;
}

Profiler & Profiler::getInstance() {
	static Profiler instance;
	return instance;
}

void Profiler::init(std::string str) {
	fs = new std::ofstream;
	fs->open(str, std::ios::app);
	partition_time = 0;
	kernel_time = 0;
	count = 0;
}

void Profiler::addTime(float ptime, float ktime, float time) {
	partition_time += ptime;
	kernel_time += ktime;
	this->time += time;
	count++;
}

std::ofstream & Profiler::getFs() {
	return *fs;
}