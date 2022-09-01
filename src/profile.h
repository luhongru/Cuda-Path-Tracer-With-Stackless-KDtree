#pragma once
#include <fstream>


class Profiler {
	std::ofstream* fs;
	Profiler();
	float partition_time;
	float kernel_time;
	float time;
	float count;

public:
	void init(std::string str);
	void addTime(float ptime, float ktime, float time);
	static Profiler& getInstance();
	std::ofstream & getFs();
	~Profiler();
	int triCount;
};