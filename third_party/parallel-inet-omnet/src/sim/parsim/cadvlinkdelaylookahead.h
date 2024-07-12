//=========================================================================
//  CLINKDELAYLOOKAHEAD.H - part of
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

#ifndef __CADVLINKDELAYLOOKAHEAD_H__
#define __CADVLINKDELAYLOOKAHEAD_H__

#include "cnmplookahead.h"

NAMESPACE_BEGIN

/**
 * Lookahead calculation based on inter-partition link delays only.
 *
 * @ingroup Parsim
 */
class SIM_API cAdvancedLinkDelayLookahead : public cNMPLookahead
{
  protected:
    struct LinkOut
    {
        simtime_t lookahead; // lookahead on this link (currently the link delay)
        simtime_t eot;       // current EOT on this link (last msg sent + lookahead)
    };
    struct PartitionInfo
    {
        int numLinks;        // size of links[] array
        LinkOut **links;     // information on outgoing links (needed for EOT calculation)
        simtime_t lookahead; // lookahead to partition (minimum of all link lookaheads)
    };

    // partition information
    int numSeg;              // number of partitions
    PartitionInfo *segInfo;  // partition info array, size numSeg

  public:
    /**
     * Constructor.
     */
    cAdvancedLinkDelayLookahead();

    /**
     * Destructor.
     */
    virtual ~cAdvancedLinkDelayLookahead();

    /**
     * Sets up algorithm for new simulation run.
     */
    virtual void startRun();

    /**
     * Called at end of simulation run.
     */
    virtual void endRun();

    /**
     * Updates lookahead information, based on the delay of the link
     * where message is sent out. Returns EOT.
     */
    virtual simtime_t getCurrentLookahead(cMessage *msg, int procId, void *data);

    /**
     * Returns minimum of link delays toward the given partition.
     */
    virtual simtime_t getCurrentLookahead(int procId);
};

NAMESPACE_END


#endif


