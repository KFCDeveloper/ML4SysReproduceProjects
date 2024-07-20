//==========================================================================
//  TKLIB.CC - part of
//
//                     OMNeT++/OMNEST
//            Discrete System Simulation in C++
//
//==========================================================================

/*--------------------------------------------------------------*
  Copyright (C) 1992-2008 Andras Varga
  Copyright (C) 2006-2008 OpenSim Ltd.

  This file is distributed WITHOUT ANY WARRANTY. See the file
  `license' for details on this and other legal matters.
*--------------------------------------------------------------*/

// Work around an MSVC .NET bug: WinSCard.h causes compilation
// error due to faulty LPCVOID declaration, so prevent it from
// being pulled in (we don't need the SmartCard API here anyway ;-)
#define _WINSCARD_H_

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <assert.h>

#include "tklib.h"
#include "tkutil.h"
#include "tkenv.h"
#include "exception.h"

NAMESPACE_BEGIN

int exitOmnetpp;
extern "C" int Tkpng_Init(Tcl_Interp *interp);

// Procedure to handle X errors
static int XErrorProc(ClientData, XErrorEvent *errEventPtr)
{
    fprintf(stderr, "X protocol error: ");
    fprintf(stderr, "error=%d request=%d minor=%d\n",
                    errEventPtr->error_code,
                    errEventPtr->request_code,
                    errEventPtr->minor_code );
    return 0;  // claim to have handled the error
}

// initialize Tcl/Tk and return a pointer to the interpreter
Tcl_Interp *initTk(int argc, char **argv)
{
    // create interpreter
    Tcl_Interp *interp = Tcl_CreateInterp();

    // pass application name to interpreter
    Tcl_FindExecutable(argv[0]);

    // Tcl/Tk args interfere with OMNeT++'s own command-line args
    //if (Tk_ParseArgv(interp, (Tk_Window)NULL, &argc, argv, argTable, 0)!=TCL_OK)
    //{
    //    fprintf(stderr, "%s\n", Tcl_GetStringResult(interp));
    //    return TCL_ERROR;
    //}

    if (Tcl_Init(interp) != TCL_OK)
        throw opp_runtime_error("Tkenv: Tcl_Init failed: %s\n", Tcl_GetStringResult(interp));

    if (Tk_Init(interp) != TCL_OK)
        throw opp_runtime_error("Tkenv: Tk_Init failed: %s\n", Tcl_GetStringResult(interp));

    Tcl_StaticPackage(interp, "Tk", Tk_Init, (Tcl_PackageInitProc *) NULL);

    if (Tkpng_Init(interp) != TCL_OK)
        throw opp_runtime_error("Tkenv: Tkpng_Init failed: %s\n", Tcl_GetStringResult(interp));


    Tk_Window mainWindow = Tk_MainWindow(interp);

    Tk_SetAppName( mainWindow, "omnetpp" );
    Tk_SetClass( mainWindow, "Omnetpp" );

    // Register X error handler and ask for synchronous protocol to help debugging
    Tk_CreateErrorHandler( Tk_Display(mainWindow), -1,-1,-1, XErrorProc, (ClientData)mainWindow );

    // Grab initial size and background
    Tk_GeometryRequest(mainWindow,200,200);

    return interp;
}

// create custom commands (implemented in tkcmd.cc) in Tcl
int createTkCommands(Tcl_Interp *interp, OmnetTclCommand *commands)
{
    for (; commands->namestr!=NULL; commands++)
    {
        Tcl_CreateCommand( interp, TCLCONST(commands->namestr),
                                   (Tcl_CmdProc *)commands->func,
                                   (ClientData)NULL,
                                   (Tcl_CmdDeleteProc *)NULL);
    }
    return TCL_OK;
}

// run the Tk application
int runTk(Tcl_Interp *)
{
    // Custom event loop
    //  the C++ variable exitOmnetpp is used for exiting
    while (!exitOmnetpp)
    {
       Tk_DoOneEvent(TK_ALL_EVENTS);
    }

    return TCL_OK;
}

NAMESPACE_END
