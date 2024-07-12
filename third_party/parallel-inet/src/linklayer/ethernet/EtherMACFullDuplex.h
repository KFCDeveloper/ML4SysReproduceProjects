//
// Copyright (C) 2006 Levente Meszaros
// Copyright (C) 2014 RWTH Aachen University, Chair of Communication and Distributed Systems
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program; if not, see <http://www.gnu.org/licenses/>.
//

#ifndef __INET_ETHER_DUPLEX_MAC_H
#define __INET_ETHER_DUPLEX_MAC_H

#include "INETDefs.h"

#include "EtherMACBase.h"
#include "EtherMACState.h"

/**
 * A simplified version of EtherMAC. Since modern Ethernets typically
 * operate over duplex links where's no contention, the original CSMA/CD
 * algorithm is no longer needed. This simplified implementation doesn't
 * contain CSMA/CD, frames are just simply queued up and sent out one by one.
 */
class INET_API EtherMACFullDuplex : public EtherMACBase
{
  public:
    EtherMACFullDuplex();

  protected:
    virtual cState& initialize(int stage, cState &state);
    virtual int numInitStages() const  {return 3;}
    virtual void initializeStatistics();
    virtual void initializeFlags();
    virtual void initializeMACAddress();
    virtual void handleMessage(cMessage *msg);

    // finish
    virtual void finish();

    // event handlers
    virtual void handleEndIFGPeriod();
    virtual void handleEndTxPeriod();
    virtual void handleEndPausePeriod();
    virtual void handleSelfMessage(cMessage *msg);

    // helpers
    virtual void startFrameTransmission();
    virtual void processFrameFromUpperLayer(EtherFrame *frame);
    virtual void processMsgFromNetwork(EtherTraffic *msg);
    virtual void processReceivedDataFrame(EtherFrame *frame);
    virtual void processPauseCommand(int pauseUnits);
    virtual void scheduleEndIFGPeriod();
    virtual void scheduleEndPausePeriod(int pauseUnits);
    virtual void beginSendFrames();


    // statistics
    simtime_t totalSuccessfulRxTime; // total duration of successful transmissions on channel
    simtime_t totalSuccessfulTxTime; //*
    //internal
    EtherMACState *currentState;

    protected: //*
    // if a homaPkt is being sent or received, this function counts number of
    // bytes in that packet.
    virtual void countHomaPktBytes(EtherFrame* curFrame, bool isSent);
};


#endif

