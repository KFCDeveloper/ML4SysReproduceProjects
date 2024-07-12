//=========================================================================
//  MESSAGETAGS.H - part of
//
//                  OMNeT++/OMNEST
//           Discrete System Simulation in C++
//
//  Author: Andras Varga, 2003
//          Dept. of Electrical and Computer Systems Engineering,
//          Monash University, Melbourne, Australia
//
//=========================================================================

/*--------------------------------------------------------------*
  Copyright (C) 2003-2008 Andras Varga
  Copyright (C) 2006-2008 OpenSim Ltd.

  This file is distributed WITHOUT ANY WARRANTY. See the file
  `license' for details on this and other legal matters.
*--------------------------------------------------------------*/

#ifndef __MESSAGETAGS_H
#define __MESSAGETAGS_H

//
// message tags:
//
enum {
     TAG_SETUP_LINKS = 1000,
     TAG_RUNNUMBER,
     TAG_CMESSAGE,
     TAG_NULLMESSAGE,
     TAG_CMESSAGE_WITH_NULLMESSAGE,
     TAG_TERMINATIONEXCEPTION,
     TAG_EXCEPTION,
     TAG_PARALLELINITIALIZATION,
     TAG_SETUP_MODULE_PROC_IDS
};

#endif
