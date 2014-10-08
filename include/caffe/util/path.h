#ifndef _PATH_H_
#define _PATH_H_

#ifdef WIN32
#include <stdlib.h>
#ifndef MAX_PATH
#define MAX_PATH _MAX_PATH
#endif
#else
#include <libgen.h>
#ifndef MAX_PATH
#define MAX_PATH 2048
#endif
#endif // WIN32

#include <string>
using std::string;

class CPath
{

public:

	static string GetDirectoryName(string strPath)
	{
#ifdef WIN32
		char pDrive[MAX_PATH], pDir[MAX_PATH], pFileName[MAX_PATH], pExt[MAX_PATH];
		char pOutput[MAX_PATH];
		::_splitpath_s(strPath.c_str(), pDrive, MAX_PATH, pDir, MAX_PATH, pFileName, MAX_PATH, pExt, MAX_PATH);
		sprintf_s(pOutput, MAX_PATH, "%s%s", pDrive, pDir);
		return string(pOutput);
#else
		char *bname;
		bname = basename(strPath);
		return string(bname);
#endif // WIN32
	}


};


#endif
