/*!re2c re2c:flags:i = 1; */
#include <stddef.h>

/*!max:re2c*/
/*!re2c
	ws		= " "*;
	ws_brack= [ \[\]]*;
	id		= [A-Za-z_] [A-Za-z0-9_-]*;
    digit	= [0-9];
    integer	= [-]? digit+;
    real	= [-]? digit+ "."? digit*;
	EQ		= "=";
*/

char* parse_header(char *YYCURSOR, char** _id, char** _kind, char** _type) {
    char *YYMARKER;
	char *id, *id2, *kind, *kind2, *type, *type2, *end;
	/*!stags:re2c format = 'char *@@;'; */

    /*!re2c
    re2c:define:YYCTYPE = char;
    re2c:yyfill:enable  = 0;

    * { return NULL; }

	@id id @id2
	ws
	EQ
	ws
	@kind id @kind2
	ws
	@type id @type2
	ws
	@end
	{ 
		*_id = id;
		*id2 = '\0';

		*_kind = kind;
		*kind2 = '\0';

		*_type = type;
		*type2 = '\0';

		return end;
	}
	*/
}

char* parse_int(char *YYCURSOR, char** _number) {
    char *YYMARKER;
	char *number, *number2, *end;
	/*!stags:re2c format = 'char *@@;'; */

    /*!re2c
    re2c:define:YYCTYPE = char;
    re2c:yyfill:enable  = 0;

    * { return NULL; }

	ws_brack @number integer @number2 ws_brack @end { 
		*_number = number;
		*number2 = '\0';
		return end;
	}
	*/
}

char* parse_float(char *YYCURSOR, char** _number) {
    char *YYMARKER;
	char *number, *number2, *end;
	/*!stags:re2c format = 'char *@@;'; */

    /*!re2c
    re2c:define:YYCTYPE = char;
    re2c:yyfill:enable  = 0;

    * { return NULL; }

	ws_brack @number real @number2 ws_brack @end { 
		*_number = number;
		*number2 = '\0';
		if(number2[-1] == '.')
			number2[-1] = '\0';
			
		return end;
	}
	*/
}

char* parse_id(char *YYCURSOR, char** _id) {
    char *YYMARKER;
	char *id, *id2, *end;
	/*!stags:re2c format = 'char *@@;'; */

    /*!re2c
    re2c:define:YYCTYPE = char;
    re2c:yyfill:enable  = 0;

    * { return NULL; }

	ws @id id @id2 ws @end { 
		*_id = id;
		*id2 = '\0';
		return end;
	}
	*/
}
