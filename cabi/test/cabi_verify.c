#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

typedef char *(*fn_version)(void);
typedef void (*fn_free)(char *);

int main(int argc, char **argv) {
	const char *lib = argc > 1 ? argv[1] : "./welvet.so";
	void *h = dlopen(lib, RTLD_NOW);
	if (!h) {
		fprintf(stderr, "dlopen %s: %s\n", lib, dlerror());
		return 1;
	}
	fn_version ver = (fn_version)dlsym(h, "WelvetEngineVersion");
	fn_free freefn = (fn_free)dlsym(h, "FreeWelvetString");
	if (!ver || !freefn) {
		fprintf(stderr, "missing WelvetEngineVersion / FreeWelvetString\n");
		return 1;
	}
	char *v = ver();
	if (!v) {
		fprintf(stderr, "null version\n");
		return 1;
	}
	printf("WelvetEngineVersion=%s\n", v);
	int ok = strcmp(v, "1.1.1") == 0;
	freefn(v);
	if (!ok) {
		fprintf(stderr, "expected 1.1.1\n");
		return 1;
	}
	printf("cabi_verify OK\n");
	dlclose(h);
	return 0;
}
