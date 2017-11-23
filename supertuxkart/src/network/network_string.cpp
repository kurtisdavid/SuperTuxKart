//  SuperTuxKart - a fun racing game with go-kart
//  Copyright (C) 2014-2015 SuperTuxKart-Team
//
//  This program is free software; you can redistribute it and/or
//  modify it under the terms of the GNU General Public License
//  as published by the Free Software Foundation; either version 3
//  of the License, or (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

#include "network/network_string.hpp"

#include "utils/string_utils.hpp"

#include <algorithm>   // for std::min
#include <iomanip>
#include <ostream>

// ============================================================================
/** Unit testing function.
 */
void NetworkString::unitTesting()
{
    NetworkString s(PROTOCOL_LOBBY_ROOM);
    assert(s.getProtocolType() == PROTOCOL_LOBBY_ROOM);
    assert(s.getProtocolType() != PROTOCOL_KART_UPDATE);
    assert(!s.isSynchronous());
    s.setSynchronous(true);
    assert(s.isSynchronous());
    s.setSynchronous(false);
    assert(!s.isSynchronous());

    uint32_t token = 0x12345678;
    // Check token setting and reading
    s.setToken(token);
    assert(s.getToken()==token);
    assert(s.getToken()!=0x87654321);

    // Append some values from the message
    s.addUInt16(12345);
    s.addFloat(1.2345f);

    // Since this string was not received, we need to skip the type and token explicitly.
    s.skip(5);
    assert(s.getUInt16() == 12345);
    float f = s.getFloat();
    assert(f==1.2345f);

    // Check modifying a token in an already assembled message
    uint32_t new_token = 0x87654321;
    s.setToken(new_token);
    assert(s.getToken()!=token);
    assert(s.getToken()==new_token);

    // Check log message format
    BareNetworkString slog(28);
    for(unsigned int i=0; i<28; i++)
        slog.addUInt8(i);
    std::string log = slog.getLogMessage();
    assert(log=="0x000 | 00 01 02 03 04 05 06 07  08 09 0a 0b 0c 0d 0e 0f   | ................\n"
                "0x010 | 10 11 12 13 14 15 16 17  18 19 1a 1b               | ............\n");
}   // unitTesting

// ============================================================================

// ----------------------------------------------------------------------------
/** Adds one byte for the length of the string, and then (up to 255 of)
 *  the characters of the given string. */
BareNetworkString& BareNetworkString::encodeString(const std::string &value)
{
    int len = (int)value.size();
    if(len<=255)
        return this->addUInt8(len).addString(value);
    else
        return addUInt8(255).addString(value.substr(0, 255));
}   // encodeString

// ----------------------------------------------------------------------------
 /** Adds one byte for the length of the string, and then (up to 255 of)
 *  the characters of the given string. */
BareNetworkString& BareNetworkString::encodeString(const irr::core::stringw &value)
{
    std::string v = StringUtils::wideToUtf8(value);
    return encodeString(v);
}   // encodeString

// ----------------------------------------------------------------------------
/** Returns a string at the given position. The first byte indicates the
 *  length, followed by the actual string (not 0 terminated).
 *  \param[in] pos Buffer position where the encoded string starts.
 *  \param[out] out The decoded string.
 *  \return number of bytes read = 1+length of string
 */
int BareNetworkString::decodeString(std::string *out) const
{
    uint8_t len = get<uint8_t>();
    *out = getString(len);
    return len+1;
}    // decodeString

// ----------------------------------------------------------------------------
/** Returns an irrlicht wide string from the utf8 encoded string at the 
 *  given position.
 *  \param[out] out The decoded string.
 *  \return number of bytes read. If there are no special characters in the
 *          string that will be 1+length of string, but multi-byte encoded
 *          characters can mean that the length of the returned string is
 *          less than the number of bytes read.
 */
int BareNetworkString::decodeStringW(irr::core::stringw *out) const
{
    std::string s;
    int len = decodeString(&s);
    *out = StringUtils::utf8ToWide(s);
    return len;
}   // decodeString 

// ----------------------------------------------------------------------------
/** Returns a string representing this message suitable to be printed
 *  to stdout or via the Log mechanism. Format
 *   0000 : 1234 5678 9abc  ...    ASCII-
 */
std::string BareNetworkString::getLogMessage(const std::string &indent) const
{
    std::ostringstream oss;
    for(unsigned int line=0; line<m_buffer.size(); line+=16)
    {
        oss << "0x" << std::hex << std::setw(3) << std::setfill('0') 
            << line << " | ";
        unsigned int upper_limit = std::min(line+16, (unsigned int)m_buffer.size());
        for(unsigned int i=line; i<upper_limit; i++)
        {
            oss << std::hex << std::setfill('0') << std::setw(2) 
                << int(m_buffer[i])<< ' ';
            if(i%8==7) oss << " ";
        }   // for i
        // fill with spaces if necessary to properly align ascii columns
        for(unsigned int i=upper_limit; i<line+16; i++)
        {
            oss << "   ";
            if (i%8==7) oss << " ";
        }

        // Add ascii representation
        oss << " | ";
        for(unsigned int i=line; i<upper_limit; i++)
        {
            uint8_t c = m_buffer[i];
            // Don't print tabs, and characters >=128, which are often shown
            // as more than one character.
            if(isprint(c) && c!=0x09 && c<=0x80)
                oss << char(c);
            else
                oss << '.';
        }   // for i
        oss << "\n";
        // If it's not the last line, add the indentation in front
        // of the next line
        if(line+16<m_buffer.size())
            oss << indent;
    }   // for line

    return oss.str();
}   // getLogMessage

