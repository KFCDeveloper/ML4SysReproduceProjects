//==========================================================================
//  ENUMTYPE.H - part of
//                     OMNeT++/OMNEST
//            Discrete System Simulation in C++
//
//  Author: Andras Varga
//
//==========================================================================

/*--------------------------------------------------------------*
  Copyright (C) 1992-2008 Andras Varga
  Copyright (C) 2006-2008 OpenSim Ltd.

  This file is distributed WITHOUT ANY WARRANTY. See the file
  `license' for details on this and other legal matters.
*--------------------------------------------------------------*/

#ifndef __ENUMTYPE_H
#define __ENUMTYPE_H

#include <map>
#include <vector>
#include <string>
#include "scavedefs.h"

NAMESPACE_BEGIN

/**
 * Provides string representation for enums. The class basically implements
 * effective integer-to-string and string-to-integer mapping.
 */
class SCAVE_API EnumType
{
  private:
    std::map<int,std::string> valueToNameMap;
    std::map<std::string,int> nameToValueMap;

  private:
    void copy(const EnumType& other);

  public:
    /**
     * Constructor.
     */
    EnumType();

    /**
     * Copy constructor.
     */
    EnumType(const EnumType& list);

    /**
     * Destructor.
     */
    virtual ~EnumType();

    /**
     * Assignment operator. The name member is not copied;
     * see cOwnedObject's operator=() for more details.
     */
    EnumType& operator=(const EnumType& list);

    /**
     * Add an item to the enum. If that numeric code exist, overwrite it.
     */
    void insert(int value, const char *name);

    /**
     * Look up value and return string representation. Return
     * NULL if not found.
     */
    const char *nameOf(int value) const;

    /**
     * Look up string and return numeric code. If not found,
     * return the second argument.
     */
    int valueOf(const char *name, int fallback=-1) const;

    /**
     * Returns the names of the enum members sorted by their values.
     */
    std::vector<std::string> getNames() const;

    /**
     * String representation of the definition of this enum.
     */
    std::string str() const;

    /**
     * Parses the name/value pairs from a string.
     */
    void parseFromString(const char *str);
};

NAMESPACE_END


#endif

