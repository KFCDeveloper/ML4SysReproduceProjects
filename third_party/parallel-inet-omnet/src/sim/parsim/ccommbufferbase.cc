//=========================================================================
//  CCOMMBUFFERBASE.CC - part of
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


#include "ccommbufferbase.h"
#include "cexception.h"

NAMESPACE_BEGIN


cCommBufferBase::cCommBufferBase()
{
    mBuffer = 0;
    mBufferSize = 0;
    mPosition = 0;
    mMsgSize = 0;
}

cCommBufferBase::~cCommBufferBase ()
{
    delete [] mBuffer;
}

char *cCommBufferBase::getBuffer() const
{
    return mBuffer;
}

int cCommBufferBase::getBufferLength() const
{
    return mBufferSize;
}

void cCommBufferBase::allocateAtLeast(int size)
{
    size += 4; // allocate a bit more room, for sentry (used in cFileCommBuffer, etc.)
    if (mBufferSize < size)
    {
        delete [] mBuffer;
        mBuffer = new char[size];
        mBufferSize = size;
    }
}

void cCommBufferBase::setMessageSize(int size)
{
    mMsgSize = size;
    mPosition = 0;
}

int cCommBufferBase::getMessageSize() const
{
    return mMsgSize;
}

void cCommBufferBase::reset()
{
    mMsgSize = 0;
    mPosition = 0;
}

void cCommBufferBase::extendBufferFor(int dataSize)
{
    // TBD move reallocate+copy out of loop (more efficient)
    while (mMsgSize+dataSize >= mBufferSize)
    {
        // increase the size of the buffer while
        // retaining its own existing contents
        char *tempBuffer;
        int i, oldBufferSize = 0;

        oldBufferSize = mBufferSize;
        if (mBufferSize == 0)
            mBufferSize = 1000;
        else
            mBufferSize += mBufferSize;

        tempBuffer = new char[mBufferSize];
        for(i = 0; i < oldBufferSize; i++)
            tempBuffer[i] = mBuffer[i];

        delete [] mBuffer;
        mBuffer = tempBuffer;
    }
}

bool cCommBufferBase::isBufferEmpty() const
{
    return mPosition == mMsgSize;
}

void cCommBufferBase::assertBufferEmpty()
{
    if (mPosition == mMsgSize)
        return;

    if (mPosition > mMsgSize)
    {
        throw cRuntimeError("internal error: cCommBuffer pack/unpack mismatch: "
                             "read %d bytes past end of buffer while unpacking %d bytes",
                             mPosition-mMsgSize, mPosition);
    }
    else
    {
        throw cRuntimeError("internal error: cCommBuffer pack/unpack mismatch: "
                            "%d extra bytes remained in buffer after unpacking %d bytes",
                            mMsgSize-mPosition, mPosition);
    }
}

NAMESPACE_END

