//=========================================================================
//  CPARSIMPARTITION.CC - part of
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
  Copyright (C) 2014 RWTH Aachen University, Chair of Communication and Distributed Systems

  This file is distributed WITHOUT ANY WARRANTY. See the file
  `license' for details on this and other legal matters.
*--------------------------------------------------------------*/

#include <stdlib.h>
#include <stdio.h>
#include "cmessage.h"
#include "errmsg.h"
#include "cplaceholdermod.h"
#include "cproxygate.h"
#include "cparsimpartition.h"
#include "ccommbuffer.h"
#include "cparsimcomm.h"
#include "cparsimsynchr.h"
#include "creceivedexception.h"
#include "messagetags.h"
#include "cenvir.h"
#include "cconfiguration.h"
#include "globals.h"
#include "cconfigoption.h"
#include "regmacros.h"

NAMESPACE_BEGIN

Register_Class(cParsimPartition);

Register_GlobalConfigOption(CFGID_PARSIM_DEBUG, "parsim-debug", CFG_BOOL, "true", "With parallel-simulation=true: turns on printing of log messages from the parallel simulation code.");

cParsimPartition::cParsimPartition()
{
    sim = NULL;
    comm = NULL;
    synch = NULL;
    debug = ev.getConfig()->getAsBool(CFGID_PARSIM_DEBUG);
}

cParsimPartition::~cParsimPartition()
{
}

void cParsimPartition::setContext(cSimulation *simul, cParsimCommunications *commlayer, cParsimSynchronizer *sync)
{
    sim = simul;
    comm = commlayer;
    synch = sync;
}

void cParsimPartition::startRun()
{
    connectRemoteGates();
    configurePlaceholderModules();
}

void cParsimPartition::endRun()
{
}

void cParsimPartition::shutdown()
{
    if (!comm) return; // if initialization failed

    cException e("Process has shut down");
    broadcastException(e);

    comm->shutdown();
}

void cParsimPartition::configurePlaceholderModules()
{
    cCommBuffer *buffer = comm->createCommBuffer();

    ev << "configuring placeholder modules: step 1 - broadcasting modules procids\n";
    for (int modId=0; modId<=sim->getLastModuleId(); modId++)
        {
            cModule *mod = sim->getModule(modId);
            if (mod && !mod->isPlaceholder())
            {

                buffer->pack(mod->getId());
                buffer->pack(mod->getFullPath());
                buffer->pack(mod->getName());

            }
        }
        buffer->pack(-1); // "the end"
        comm->broadcast(buffer, TAG_SETUP_MODULE_PROC_IDS);


    ev << "configuring placeholder modules: step 2 - collecting broadcasts sent by others...\n";
    for (int i=0; i<comm->getNumPartitions()-1; i++)
        {
            // receive:
            int tag, remoteProcId;
            // note: *must* filter for TAG_SETUP_LINKS here, to prevent race conditions
            if (!comm->receiveBlocking(TAG_SETUP_MODULE_PROC_IDS, buffer, tag, remoteProcId))
                throw cRuntimeError("connectRemoteGates() interrupted by user");
            ASSERT(tag==TAG_SETUP_MODULE_PROC_IDS);

            ev << "  processing msg from procId=" << remoteProcId << "...\n";

            // process what we got:
            while(true)
            {
                int remoteModId;
                char *moduleFullPath;
                char *moduleName;

                // unpack a gate -- modId==-1 indicates end of packet
                buffer->unpack(remoteModId);
                if (remoteModId==-1)
                    break;
                buffer->unpack(moduleFullPath);
                buffer->unpack(moduleName);

                cModule *module = simulation.getModule(remoteModId);

                if(module->isPlaceholder())
                {
                    cPlaceholderModule *pm = check_and_cast<cPlaceholderModule *>(module);
                    pm->setRemoteProcId(remoteProcId);
                    ev << "    placeholder: " << moduleName << " id: " << remoteModId << endl; //TODO zu ev
                }

                delete [] moduleFullPath;
                delete [] moduleName;
            }
            buffer->assertBufferEmpty();
        }
        ev << "  done.\n";
        comm->recycleCommBuffer(buffer);

        // verify that every placeholdermodule has a procid
          for (int modId=0; modId<=sim->getLastModuleId(); modId++)
          {
              cModule *mod = sim->getModule(modId);
              if (mod && mod->isPlaceholder())
              {
                  cPlaceholderModule *pm = check_and_cast<cPlaceholderModule*>(mod);
                  if(pm->getRemoteProcId() == -1){
                      throw cRuntimeError("Placeholder Module %s has no assigned procid (-1)",pm->getName());
                  }
              }
          }
}


void cParsimPartition::connectRemoteGates()
{
    cCommBuffer *buffer = comm->createCommBuffer();

    //
    // Step 1: broadcast list of all "normal" input gates that may have corresponding
    // proxy gates in other partitions (as they need to redirect here)
    //
    ev << "connecting remote gates: step 1 - broadcasting input gates...\n";
    for (int modId=0; modId<=sim->getLastModuleId(); modId++)
    {
        cModule *mod = sim->getModule(modId);
        if (mod && !mod->isPlaceholder())
        {
            for (cModule::GateIterator i(mod); !i.end(); i++)
            {
                cGate *g = i();
                if (g->getType()==cGate::INPUT)
                {
                    // if gate is connected to a placeholder module, in another partition that will
                    // be a local module and our module will be a placeholder with proxy gates
                    cGate *srcg = g->getPathStartGate();
                    if (srcg != g && srcg->getOwnerModule()->isPlaceholder())
                    {
                        // pack gate "address" here
                        buffer->pack(g->getOwnerModule()->getId());
                        buffer->pack(g->getId());
                        // pack gate name
                        buffer->pack(g->getOwnerModule()->getFullPath().c_str());
                        buffer->pack(g->getName());
                        buffer->pack(g->isVector() ? g->getIndex() : -1);
                    }
                }
            }
        }
    }
    buffer->pack(-1); // "the end"
    comm->broadcast(buffer, TAG_SETUP_LINKS);

    //
    // Step 2: collect info broadcast by others, and use it to fill in output cProxyGates
    //
    ev << "connecting remote gates: step 2 - collecting broadcasts sent by others...\n";
    for (int i=0; i<comm->getNumPartitions()-1; i++)
    {
        // receive:
        int tag, remoteProcId;
        // note: *must* filter for TAG_SETUP_LINKS here, to prevent race conditions
        if (!comm->receiveBlocking(TAG_SETUP_LINKS, buffer, tag, remoteProcId))
            throw cRuntimeError("connectRemoteGates() interrupted by user");
        ASSERT(tag==TAG_SETUP_LINKS);

        ev << "  processing msg from procId=" << remoteProcId << "...\n";

        // process what we got:
        while(true)
        {
            int remoteModId;
            int remoteGateId;
            char *moduleFullPath;
            char *gateName;
            int gateIndex;

            // unpack a gate -- modId==-1 indicates end of packet
            buffer->unpack(remoteModId);
            if (remoteModId==-1)
                break;
            buffer->unpack(remoteGateId);
            buffer->unpack(moduleFullPath);
            buffer->unpack(gateName);
            buffer->unpack(gateIndex);

            // find corresponding output gate (if we have it) and set remote
            // gate address to the received one
            cModule *m = sim->getModuleByPath(moduleFullPath);
            cGate *g = !m ? NULL : gateIndex==-1 ? m->gate(gateName) : m->gate(gateName,gateIndex);
            cProxyGate *pg = dynamic_cast<cProxyGate *>(g);

            ev << "    gate: " << moduleFullPath << "." << gateName;
            if (gateIndex>=0)
                ev << "["  << gateIndex << "]";
             ev << " - ";
            if (!pg)
                ev << "not here\n";
            else
                ev << "points to (procId=" << remoteProcId << " moduleId=" << remoteModId << " gateId=" << remoteGateId << ")\n";

            if (pg)
            {
                pg->setPartition(this);
                pg->setRemoteGate(remoteProcId,remoteModId,remoteGateId);
            }

            delete [] moduleFullPath;
            delete [] gateName;
        }
        buffer->assertBufferEmpty();
    }
    ev << "  done.\n";
    comm->recycleCommBuffer(buffer);

    // verify that all gates have been connected
    for (int modId=0; modId<=sim->getLastModuleId(); modId++)
    {
        cModule *mod = sim->getModule(modId);
        if (mod && mod->isPlaceholder())
        {
            for (cModule::GateIterator i(mod); !i.end(); i++)
            {
                cProxyGate *pg = dynamic_cast<cProxyGate *>(i());
                if (pg && pg->getRemoteProcId()==-1 && !pg->getPathStartGate()->getOwnerModule()->isPlaceholder())
                    throw cRuntimeError("parallel simulation error: dangling proxy gate '%s' "
                        "(module '%s' or some of its submodules not instantiated in any partition?)",
                        pg->getFullPath().c_str(), mod->getFullPath().c_str());
            }
        }
    }
}

void cParsimPartition::processOutgoingMessage(cMessage *msg, int procId, int moduleId, int gateId, void *data)
{
    if (debug) ev << "sending message '" << msg->getFullName() << "' (for T="
                  << msg->getArrivalTime() << " to procId=" << procId << ")\n";

    synch->processOutgoingMessage(msg, procId, moduleId, gateId, data);
}

void cParsimPartition::processReceivedBuffer(cCommBuffer *buffer, int tag, int sourceProcId)
{
    opp_string errmsg;
    switch (tag)
    {
        case TAG_TERMINATIONEXCEPTION:
            buffer->unpack(errmsg);
            throw cReceivedTerminationException(sourceProcId, errmsg.c_str());
        case TAG_EXCEPTION:
            buffer->unpack(errmsg);
            throw cReceivedException(sourceProcId, errmsg.c_str());
        default:
            throw cRuntimeError("cParsimPartition::processReceivedBuffer(): unexpected tag %d "
                                 "from procId %d", tag, sourceProcId);
    }
    buffer->assertBufferEmpty();
}

void cParsimPartition::processReceivedMessage(cMessage *msg, int destModuleId, int destGateId, int sourceProcId)
{
    msg->setSrcProcId(sourceProcId);
    cModule *mod = sim->getModule(destModuleId);
    if (!mod)
        throw cRuntimeError("parallel simulation error: destination module id=%d for message \"%s\""
                             "from partition %d does not exist (any longer)",
                             destModuleId, msg->getName(), sourceProcId);
    cGate *g = mod->gate(destGateId);
    if (!g)
        throw cRuntimeError("parallel simulation error: destination gate %d of module id=%d "
                             "for message \"%s\" from partition %d does not exist",
                             destGateId, destModuleId, msg->getName(), sourceProcId);

    // do our best to set the source gate (the gate of a cPlaceholderModule)
    cGate *srcg = g->getPathStartGate();
    msg->setSentFrom(srcg->getOwnerModule(), srcg->getId(), msg->getSendingTime());

    // deliver it to the "destination" gate of the connection -- the channel
    // has already been simulated in the originating partition.
    g->deliver(msg, msg->getArrivalTime());
    ev.messageSent_OBSOLETE(msg); //FIXME change to the newer callback methods! messageSendHop() etc
}

void cParsimPartition::broadcastTerminationException(cTerminationException& e)
{
    // send TAG_TERMINATIONEXCEPTION to all partitions
    cCommBuffer *buffer = comm->createCommBuffer();
    buffer->pack(e.what());
    try
    {
        comm->broadcast(buffer, TAG_TERMINATIONEXCEPTION);
    }
    catch (std::exception&)
    {
        // swallow exceptions -- here we're not interested in them
    }
    comm->recycleCommBuffer(buffer);
}

void cParsimPartition::broadcastException(std::exception& e)
{
    // send TAG_EXCEPTION to all partitions
    cCommBuffer *buffer = comm->createCommBuffer();
    buffer->pack(e.what());
    try
    {
        comm->broadcast(buffer, TAG_EXCEPTION);
    }
    catch (std::exception&)
    {
        // swallow any exceptions -- here we're not interested in them
    }
    comm->recycleCommBuffer(buffer);
}

NAMESPACE_END

