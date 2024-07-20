//==========================================================================
//   CNEDVALUE.CC  - part of
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

#include "cnedvalue.h"
#include "cxmlelement.h"
#include "cexception.h"
#include "cpar.h"
#include "stringutil.h"
#include "stringpool.h"
#include "unitconversion.h"

NAMESPACE_BEGIN

void cNEDValue::operator=(const cNEDValue& other)
{
    type = other.type;
    switch (type)
    {
        case UNDEF: break;
        case BOOL: bl = other.bl; break;
        case DBL: dbl = other.dbl; dblunit = other.dblunit; break;
        case STR: s = other.s; break;
        case XML: xml = other.xml; break;
    }
}

const char *cNEDValue::getTypeName(Type t)
{
    switch (t)
    {
        case UNDEF:  return "undef";
        case BOOL:   return "bool";
        case DBL:    return "double";
        case STR:    return "string";
        case XML:    return "xml";
        default:     return "???";
    }
}

void cNEDValue::cannotCastError(Type t) const
{
    throw cRuntimeError("cNEDValue: cannot cast %s from type %s to %s",
        str().c_str(), getTypeName(type), getTypeName(t));
}

void cNEDValue::set(const cPar& par)
{
    switch (par.getType())
    {
        case cPar::BOOL: *this = par.boolValue(); break;
        case cPar::DOUBLE: *this = par.doubleValue(); dblunit = par.getUnit(); break;
        case cPar::LONG: *this = par.doubleValue(); dblunit = par.getUnit(); break;
        case cPar::STRING: *this = par.stdstringValue(); break;
        case cPar::XML: *this = par.xmlValue(); break;
        default: throw cRuntimeError("internal error: bad cPar type: %s", par.getFullPath().c_str());
    }
}

void cNEDValue::convertTo(const char *unit)
{
    dbl = convertUnit(dbl, dblunit, unit);
    dblunit = unit;
}

double cNEDValue::convertUnit(double d, const char *unit, const char *targetUnit)
{
    // not inline because simkernel header files cannot refer to common/ headers (unitconversion.h)
    return UnitConversion::convertUnit(d, unit, targetUnit);
}

double cNEDValue::parseQuantity(const char *str, const char *expectedUnit)
{
    return UnitConversion::parseQuantity(str, expectedUnit);
}

double cNEDValue::parseQuantity(const char *str, std::string& outActualUnit)
{
    return UnitConversion::parseQuantity(str, outActualUnit);
}

const char *cNEDValue::getPooled(const char *s)
{
    static CommonStringPool stringPool; // non-refcounted
    return stringPool.get(s);
}

std::string cNEDValue::str() const
{
    char buf[32];
    switch (type)
    {
        case BOOL: return bl ? "true" : "false";
        case DBL:  sprintf(buf, "%g%s", dbl, opp_nulltoempty(dblunit)); return buf;
        case STR:  return opp_quotestr(s.c_str());
        case XML:  return xml->detailedInfo();
                   //or: return std::string("<")+xml->getTagName()+" ... >, " + opp_nulltoempty(xml->getSourceLocation());
        default:   throw cRuntimeError("internal error: bad cNEDValue type");
    }
}

NAMESPACE_END

