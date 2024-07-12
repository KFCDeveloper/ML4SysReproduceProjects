//=========================================================================
//  CLINKDELAYLOOKAHEAD.CC - part of
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


#include "cadvlinkdelaylookahead.h"
#include "csimulation.h"
#include "cmessage.h"
#include "cenvir.h"
#include "cnullmessageprot.h"
#include "cparsimcomm.h"
#include "cparsimpartition.h"
#include "cplaceholdermod.h"
#include "cproxygate.h"
#include "cchannel.h"
#include "globals.h"
#include "regmacros.h"

NAMESPACE_BEGIN


Register_Class(cAdvancedLinkDelayLookahead);


cAdvancedLinkDelayLookahead::cAdvancedLinkDelayLookahead()
{
    numSeg = 0;
    segInfo = NULL;
}

cAdvancedLinkDelayLookahead::~cAdvancedLinkDelayLookahead()
{
    delete [] segInfo;
}

void cAdvancedLinkDelayLookahead::startRun()
{
    ev << "starting Link Delay Lookahead...\n";

    delete [] segInfo;

    numSeg = comm->getNumPartitions();
    segInfo = new PartitionInfo[numSeg];

    // temporarily initialize everything to zero.
    for (int i=0; i<numSeg; i++)
    {
        segInfo[i].numLinks = 0;
        segInfo[i].links = NULL;
    }

    // fill numLinks and links[]
    ev << "  collecting links...\n";

    // step 1: count gates
    for (int modId=0; modId<=sim->getLastModuleId(); modId++)
    {
        cPlaceholderModule *mod = dynamic_cast<cPlaceholderModule *>(sim->getModule(modId));
        if (mod)
        {
            for (cModule::GateIterator i(mod); !i.end(); i++)
            {
                cGate *g = i();
                cProxyGate *pg  = dynamic_cast<cProxyGate *>(g);
                if (pg && pg->getPreviousGate() && pg->getRemoteProcId()>=0)
                    segInfo[pg->getRemoteProcId()].numLinks++;
            }
        }
    }

    // step 2: allocate links[]
    for (int i=0; i<numSeg; i++)
    {
        int numLinks = segInfo[i].numLinks;
        segInfo[i].links = new LinkOut *[numLinks];
        for (int k=0; k<numLinks; k++)
            segInfo[i].links[k] = NULL;
    }

    // step 3: fill in
    for (int modId=0; modId<=sim->getLastModuleId(); modId++)
    {
        cPlaceholderModule *mod = dynamic_cast<cPlaceholderModule *>(sim->getModule(modId));
        if (mod)
        {
            for (cModule::GateIterator i(mod); !i.end(); i++)
            {
                // if this is a properly connected proxygate, process it
                // FIXME leave out gates from other cPlaceholderModules
                cGate *g = i();
                cProxyGate *pg  = dynamic_cast<cProxyGate *>(g);
                if (pg && pg->getPreviousGate() && pg->getRemoteProcId()>=0)
                {
                    // check we have a delay on this link (it gives us lookahead)
                    cGate *fromg  = pg->getPreviousGate();
                    cChannel *chan = fromg ? fromg->getChannel() : NULL;
                    cDatarateChannel *datarateChan = dynamic_cast<cDatarateChannel *>(chan);
                    cPar *delaypar = datarateChan ? datarateChan->getDelay() : NULL;
                    double linkDelay = delaypar ? delaypar->doubleValue() : 0;
                    if (linkDelay<=0.0)
                        throw cRuntimeError("cAdvancedLinkDelayLookahead: zero delay on link from gate `%s', no lookahead for parallel simulation", fromg->getFullPath().c_str());

                    // store
                    int procId = pg->getRemoteProcId();
                    int k=0;
                    while (segInfo[procId].links[k]) k++; // find 1st empty slot
                    LinkOut *link = new LinkOut;
                    segInfo[procId].links[k] = link;
                    pg->setSynchData(link);
                    link->lookahead = linkDelay;
                    link->eot = 0.0;

                    ev << "    link " << k << " to procId=" << procId << " on gate `" << fromg->getFullPath() <<"': delay=" << linkDelay << "\n";
                }
            }
        }
    }

    ev << "  setup done.\n";
}

void cAdvancedLinkDelayLookahead::endRun()
{
}

simtime_t cAdvancedLinkDelayLookahead::getCurrentLookahead(cMessage *msg, int procId, void *data)
{
    // find LinkOut structure in segInfo[destProcId].
    LinkOut *link = (LinkOut *)data;
    if (!link)
        throw cRuntimeError("internal parallel simulation error: cProxyGate has no associated data pointer");

    // calculate EOT
    simtime_t eot;
    simtime_t now = sim->getSimTime();
    simtime_t newLinkEot = now + link->lookahead;

    // TBD finish...
    return 0.0;
}

simtime_t cAdvancedLinkDelayLookahead::getCurrentLookahead(int procId)
{
    return segInfo[procId].lookahead;
}

NAMESPACE_END

