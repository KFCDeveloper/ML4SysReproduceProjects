//=========================================================================
//  CNOSYNCHRONIZATION.H - part of
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

#ifndef __NOSYNCHRONIZATION_H__
#define __NOSYNCHRONIZATION_H__

#include "cparsimprotocolbase.h"

NAMESPACE_BEGIN

// forward declarations
class cCommBuffer;

/**
 * "Dummy" implementation -- just pass messages between partitions,
 * without any synchronization. Of course incausalities may occur
 * which terminate the simulation with error, so this class is
 * only useful as a base "template" for implementing "real" synchronization
 * protocols.
 *
 * @ingroup Parsim
 */
class SIM_API cNoSynchronization : public cParsimProtocolBase
{
  public:
    /**
     * Constructor.
     */
    cNoSynchronization();

    /**
     * Destructor.
     */
    virtual ~cNoSynchronization();

    /**
     * Called at the beginning of a simulation run.
     */
    virtual void startRun();

    /**
     * Called at the end of a simulation run.
     */
    virtual void endRun();

    /**
     * Scheduler function. It processes whatever comes from other partitions,
     * then returns msgQueue.peekFirst(). Performs no synchronization
     * with other partitions, so incausalities may occur which stop the
     * simulation with an error (see also class comment).
     */
    virtual cMessage *getNextEvent();
};

NAMESPACE_END


#endif

