//==========================================================================
//  GLOBALS.H - part of
//                     OMNeT++/OMNEST
//            Discrete System Simulation in C++
//
//==========================================================================

/*--------------------------------------------------------------*
  Copyright (C) 1992-2008 Andras Varga
  Copyright (C) 2006-2008 OpenSim Ltd.
  Copyright (C) 2014 RWTH Aachen University, Chair of Communication and Distributed Systems

  This file is distributed WITHOUT ANY WARRANTY. See the file
  `license' for details on this and other legal matters.
*--------------------------------------------------------------*/

#ifndef __GLOBALS_H
#define __GLOBALS_H

#include "onstartup.h"
#include "cregistrationlist.h"
#include "cobjectfactory.h"

NAMESPACE_BEGIN

//
// Global objects
//

//< Internal: list in which objects are accumulated if there is no simple module in context.
//< @see cOwnedObject::setDefaultOwner() and cSimulation::setContextModule())
SIM_API extern cDefaultList defaultList;

SIM_API extern cGlobalRegistrationList componentTypes;  ///< List of all component types (cComponentType)
SIM_API extern cGlobalRegistrationList nedFunctions;    ///< List if all NED functions (cNEDFunction and cNEDMathFunction)
SIM_API extern cGlobalRegistrationList classes;         ///< List of all classes that can be instantiated using createOne(); see cObjectFactory and Register_Class() macro
SIM_API extern cGlobalRegistrationList enums;           ///< List of all enum objects (cEnum)
SIM_API extern cGlobalRegistrationList classDescriptors;///< List of all class descriptors (cClassDescriptor)
SIM_API extern cGlobalRegistrationList configOptions;   ///< List of supported configuration options (cConfigOption)
SIM_API extern cGlobalRegistrationList resultFilters;   ///< List of result filters (cResultFilter)
SIM_API extern cGlobalRegistrationList resultRecorders; ///< List of result recorders (cResultRecorder)
SIM_API extern cGlobalRegistrationList messagePrinters; ///< List of message printers (cMessagePrinter)
SIM_API extern cGlobalRegistrationList parInitModules;  ///< List of all modules, which want to be initialized parallel.

NAMESPACE_END


#endif

