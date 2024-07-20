//=========================================================================
//  CCOMMBUFFERBASE.H - part of
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

#ifndef __CCOMMBUFFERBASE_H__
#define __CCOMMBUFFERBASE_H__

#include "ccommbuffer.h"

NAMESPACE_BEGIN

/**
 * Adds buffer (re)allocation functions to cCommBuffer. This functionality
 * is not always needed, e.g. PVM manages its pack/unpack buffers internally.
 */
class SIM_API cCommBufferBase : public cCommBuffer
{
  protected:
    char *mBuffer;    // the buffer
    int mBufferSize;  // size of buffer allocated
    int mMsgSize;     // current msg size (incremented by pack() functions)
    int mPosition;    // current position in buffer for unpacking

  protected:
    void extendBufferFor(int dataSize);

  public:
    /**
     * Constructor.
     */
    cCommBufferBase();

    /**
     * Destructor.
     */
    virtual ~cCommBufferBase();

    cCommBufferBase(const cCommBufferBase& other) : cCommBuffer(other)
    {
        mBuffer = new char[other.mBufferSize];
        memcpy(mBuffer,other.mBuffer,other.mBufferSize*sizeof(char));
        
        mBufferSize = other.mBufferSize;
        mMsgSize = other.mMsgSize;
        mPosition = other.mPosition;
     }

    /** @name Buffer management */
    //@{
    /**
     * Returns the buffer after packing.
     */
    char *getBuffer() const;

    /**
     * Returns the size of the buffer.
     */
    int getBufferLength() const;

    /**
     * Extend buffer to the given size is needed. Existing buffer contents
     * may be lost.
     */
    void allocateAtLeast(int size);

    /**
     * Set message length in the buffer. Used after receiving a message
     * and copying it to the buffer.
     */
    void setMessageSize(int size);

    /**
     * Returns message length in the buffer.
     */
    int getMessageSize() const;

    /**
     * Reset buffer to an empty state.
     */
    void reset();

    /**
     * Returns true if all data in buffer was used up during unpacking.
     * Returns false if there was underflow (too much data unpacked)
     * or still there are unpacked data in the buffer.
     */
    virtual bool isBufferEmpty() const;

    /**
     * Utility function. To be called after unpacking a communication buffer,
     * it checks whether the buffer is empty. If it is not (i.e. an underflow
     * or overflow occurred), an exception is thrown.
     */
    virtual void assertBufferEmpty();
    //@}
};

NAMESPACE_END


#endif
