#ifndef _DIRECTORY_H_
#define _DIRECTORY_H_

#ifdef WIN32
#include <windows.h>
#include <io.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif
#include <vector>
#include <string>

#ifdef WIN32
#ifndef MAX_PATH
#define MAX_PATH _MAX_PATH
#endif
#else
#ifndef MAX_PATH
#define MAX_PATH 2048
#endif
#endif

using std::vector;
using std::string;

class CDirectory
{
public:

#ifdef WIN32
	static bool Exist(const char *strPath)
	{
		return (_access(strPath, 0) == 0);
	}
#endif // WIN32


	static bool CreateDirectory(const char *strPath)
	{
#ifdef WIN32
		if (Exist(strPath))
			return false;
		char strFolder[MAX_PATH] = {0};
		size_t len = strlen(strPath);
		for (size_t i = 0; i <= len; i++)
		{
			if (strPath[i] == '\\' || strPath[i] == '/' || strPath[i] == '\0')
			{
				if (!Exist(strFolder))
				{
					if(::CreateDirectoryA(strFolder, NULL) == 0)
						return false;
				}
			}
			strFolder[i] = strPath[i];
		}
		return true;
#else
		mkdir(strPath, 0755);
		return true;
#endif
	}
};

#endif