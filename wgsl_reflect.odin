package wgsl_reflect

import "base:runtime"
import "core:fmt"
import "core:mem"
import "core:slice"
import "core:strconv"
import "core:strings"
import "vendor:wgpu"

Any_Err :: union {
	Unexpected_Char_Err,
	Unexpected_Token_Err,
	Unexpected_Value_Err,
}

Unexpected_Char_Err :: struct {
	msg:  string,
	line: int,
}

Unexpected_Token_Err :: struct {
	msg:  string,
	line: int,
	loc:  runtime.Source_Code_Location,
}

Unexpected_Value_Err :: struct {
	msg:  string,
	line: int,
	loc:  runtime.Source_Code_Location,
}

/*
~API
*/

process :: proc(
	shader: string,
	allocator := context.allocator,
) -> (
	Metadata,
	[]Any_Stmt,
	[]Token,
	[]Any_Err,
) {
	tokens, scanner_errs := scan(shader, allocator)
	statements, parser_errs := parse(tokens, allocator)
	refl, reflection_errors := reflect(statements, allocator)

	all_errs := slice.concatenate(
		[][]Any_Err{scanner_errs, parser_errs, reflection_errors},
		allocator,
	)
	return refl, statements, tokens, all_errs
}

scan :: proc(source: string, allocator := context.allocator) -> ([]Token, []Any_Err) {
	s := Scanner {
		line      = 1,
		source    = source,
		allocator = allocator,
		tokens    = make([dynamic]Token, allocator),
		errors    = make([dynamic]Any_Err, allocator),
	}

	for !s_is_at_end(&s) {
		s.start = s.current
		s_scan_token(&s)
	}

	s_add_token(&s, .EOF)
	return s.tokens[:], s.errors[:]
}

parse :: proc(tokens: []Token, allocator := context.allocator) -> ([]Any_Stmt, []Any_Err) {
	p := Parser {
		tokens    = tokens,
		allocator = allocator,
	}

	statements := make([dynamic]Any_Stmt, p.allocator)
	for !p_is_at_end(&p) {
		statement := p_global_decl_or_directive(&p)
		if statement != nil {
			append(&statements, statement)
		}
	}

	return statements[:], p.errors[:]
}

reflect :: proc(statements: []Any_Stmt, allocator := context.allocator) -> (Metadata, []Any_Err) {
	resources := make(map[string]Resource_Info, allocator)
	entry_points := make([dynamic]Function_Info, allocator)
	errors := make([dynamic]Any_Err, allocator)

	for statement in statements {
		switch s in statement {
		case ^Var_Decl:
			group, has_group := r_get_attribute_value_int(s.attributes, "group")
			binding, has_binding := r_get_attribute_value_int(s.attributes, "binding")
			is_resource := has_group || has_binding
			if !is_resource do continue

			stages, has_stages := r_get_attribute_values(s.attributes, "stage")
			if !has_group {
				append(
					&errors,
					Unexpected_Value_Err {
						msg = fmt.aprintf("Missing '@group' attribute for resource '%v'", s.name),
						line = s.line,
					},
				)
				continue
			}
			if !has_binding {
				append(
					&errors,
					Unexpected_Value_Err {
						msg = fmt.aprintf(
							"Missing '@binding' attribute for resource '%v'",
							s.name,
						),
						line = s.line,
					},
				)
				continue
			}

			visibility := wgpu.ShaderStageFlags{}
			for stage in stages {
				switch stage {
				case "vertex":
					visibility += {.Vertex}
				case "fragment":
					visibility += {.Fragment}
				case "compute":
					visibility += {.Compute}
				case:
					append(
						&errors,
						Unexpected_Value_Err {
							msg = fmt.aprintf("Unexpected '@stage' attribute value: '%v'", stage),
							line = s.line,
						},
					)
				}
			}

			info := Resource_Info {
				name       = s.name,
				group      = group,
				binding    = binding,
				visibility = visibility,
			}

			switch {
			case strings.contains(s.type.name, "texture"):
				info.type = .Texture
				switch s.type.format.name {
				case "f32":
					info.texture.sampleType = .Float
				case "i32":
					info.texture.sampleType = .Sint
				case "u32":
					info.texture.sampleType = .Uint
				}

				switch s.type.name {
				case "texture_1d":
					info.texture.viewDimension = ._1D
				case "texture_2d":
					info.texture.viewDimension = ._2D
				case "texture_2d_array":
					info.texture.viewDimension = ._2DArray
				case "texture_3d":
					info.texture.viewDimension = ._3D
				case "texture_cube":
					info.texture.viewDimension = .Cube
				case "texture_cube_array":
					info.texture.viewDimension = .CubeArray
				case "texture_multisampled_2d":
					info.texture.viewDimension = ._2D
					info.texture.multisampled = true
				case "texture_depth_multisampled_2d":
					info.texture.viewDimension = ._2D
					info.texture.multisampled = true
				case "texture_depth_2d":
					info.texture.viewDimension = ._2D
				case "texture_depth_2d_array":
					info.texture.viewDimension = ._2DArray
				case "texture_depth_cube":
					info.texture.viewDimension = .Cube
				case "texture_depth_cube_array":
					info.texture.viewDimension = .CubeArray
				}
			case strings.contains(s.type.name, "sampler"):
				switch s.type.name {
				case "sampler":
					info.type = .Sampler
					info.sampler.type = .Filtering
				case "sampler_comparison":
					info.type = .Sampler
					info.sampler.type = .Comparison
				}
			case:
				info.type = .Buffer
				#partial switch s.space {
				case .Storage:
					if s.access == .Read {
						info.buffer.type = .ReadOnlyStorage
					} else {
						info.buffer.type = .Storage
					}
				case .Uniform:
					info.buffer.type = .Uniform
				}
			}


			resources[s.name] = info
		case ^Struct_Decl:
		case ^Function_Decl:
			if r_is_entry_point(s) {
				info := Function_Info {
					name       = strings.clone_to_cstring(s.name, allocator),
					arguments  = slice.clone(s.arguments, allocator), // Clone as statements could be dealloc separately
					attributes = slice.clone(s.attributes, allocator),
				}

				if r_has_attribute(s.attributes, "vertex") {
					info.stage = .Vertex
				} else if r_has_attribute(s.attributes, "fragment") {
					info.stage = .Fragment
				} else if r_has_attribute(s.attributes, "compute") {
					info.stage = .Compute
				}

				append(&entry_points, info)
			}
		}
	}

	return {resources = resources, entry_points = entry_points[:]}, errors[:]
}

get_entry_point :: proc(m: Metadata, stage: wgpu.ShaderStage) -> (Function_Info, bool) {
	for e in m.entry_points {
		if e.stage == stage do return e, true
	}

	return {}, false
}

generate_layout_desc :: proc(
	m: Metadata,
	group_slot: int,
	allocator := context.allocator,
) -> wgpu.BindGroupLayoutDescriptor {
	assert(group_slot <= 3)
	entries := make([dynamic]wgpu.BindGroupLayoutEntry, allocator)
	for name, resource in m.resources {
		if resource.group == group_slot {
			append(
				&entries,
				wgpu.BindGroupLayoutEntry {
					binding = u32(resource.binding),
					visibility = resource.visibility,
					buffer = resource.buffer,
					sampler = resource.sampler,
					texture = resource.texture,
				},
			)
		}
	}

	return {entryCount = len(entries), entries = raw_data(entries)}
}

comply :: proc(shader: string, allocator := context.allocator) -> cstring {
	return strings.clone_to_cstring(
		strings.concatenate(
			strings.split_multi(
				shader,
				{
					"@stage(vertex) ",
					"@stage(fragment) ",
					"@stage(vertex, fragment) ",
					"@stage(compute) ",
				},
			),
			allocator,
		),
		allocator,
	)
}

/*
~Scanner
*/

Token :: struct {
	kind: Token_Kind,
	text: string,
	line: int,
}

Token_Kind :: enum {
	EOF = 0,
	NUMBER,
	STRING,
	ATTR,
	PAREN_LEFT,
	PAREN_RIGHT,
	BRACE_LEFT,
	BRACE_RIGHT,
	BRACKET_LEFT,
	BRACKET_RIGHT,
	COLON,
	SEMICOLON,
	DOT,
	COMMA,
	MINUS,
	PLUS,
	STAR,
	SLASH,
	EQUAL,
	EQUAL_EQUAL,
	BANG_EQUAL,
	BANG,
	LESS,
	LESS_EQUAL,
	GREATER,
	GREATER_EQUAL,
	AND,
	OR,
	IF,
	ELSE,
	FOR,
	FUNCTION,
	PRIVATE,
	WORKGROUP,
	UNIFORM,
	STORAGE,
	RETURN,
	VAR,
	LET,
	CONST,
	FALSE,
	TRUE,
	STRUCT,
	IDENTIFIER,
	NIL,
	READ,
	WRITE,
	READ_WRITE,
	RETURN_TYPE,
	WHILE,
}

@(private = "package")
Scanner :: struct {
	source:    string,
	tokens:    [dynamic]Token,
	start:     int,
	current:   int,
	line:      int,
	errors:    [dynamic]Any_Err,
	allocator: mem.Allocator,
}

@(private = "package")
s_scan_token :: proc(s: ^Scanner) {
	char := s_advance(s)
	switch char {
	case "(":
		s_add_token(s, .PAREN_LEFT)
	case ")":
		s_add_token(s, .PAREN_RIGHT)
	case "[":
		s_add_token(s, .BRACKET_LEFT)
	case "]":
		s_add_token(s, .BRACKET_RIGHT)
	case "{":
		s_add_token(s, .BRACE_LEFT)
	case "}":
		s_add_token(s, .BRACE_RIGHT)
	case ",":
		s_add_token(s, .COMMA)
	case ".":
		s_add_token(s, .DOT)
	case "-":
		s_add_token(s, s_match(s, ">") ? .RETURN_TYPE : .MINUS)
	case "+":
		s_add_token(s, .PLUS)
	case ";":
		s_add_token(s, .SEMICOLON)
	case ":":
		s_add_token(s, .COLON)
	case "*":
		s_add_token(s, .STAR)
	case "!":
		s_add_token(s, s_match(s, "=") ? .BANG_EQUAL : .BANG)
	case "=":
		s_add_token(s, s_match(s, "=") ? .EQUAL_EQUAL : .EQUAL)
	case "<":
		s_add_token(s, s_match(s, "=") ? .LESS_EQUAL : .LESS)
	case ">":
		s_add_token(s, s_match(s, "=") ? .GREATER_EQUAL : .GREATER)
	case "/":
		if s_match(s, "/") {
			for s_peek(s) != "\n" && !s_is_at_end(s) {
				s_advance(s)
			}
		} else {
			s_add_token(s, .SLASH)
		}
	case "@":
		s_add_token(s, .ATTR)
	case "&": // Skip
	case " ":
		fallthrough
	case "\t":
		fallthrough
	case "\r":
	// Ignore whitespace
	case "\n":
		s.line += 1
	case:
		switch {
		case is_digit(char):
			for is_digit(s_peek(s)) do s_advance(s)

			// Look for a fractional part
			if s_peek(s) == "." && is_digit(s_peek_next(s)) {
				s_advance(s)

				for is_digit(s_peek(s)) do s_advance(s)
			}

			s_add_token(
				s,
				.NUMBER,
				//strconv.parse_f32(substring(sc.source, sc.start, sc.current)),
			)
		case is_alpha(char):
			for is_alpha_numeric(s_peek(s)) do s_advance(s)

			str := substring(s.source, s.start, s.current)
			kind := s_check_keyword(str)
			s_add_token(s, kind)
		case:
			append(
				&s.errors,
				Unexpected_Char_Err {
					msg = fmt.aprintf("Unknown char: %v", char, allocator = s.allocator),
					line = s.line,
				},
			)
		}
	}
}

@(private = "package")
s_advance :: proc(s: ^Scanner) -> string {
	char := char_at(s.source, s.current)
	s.current += 1
	return char
}

@(private = "package")
s_add_token :: proc(s: ^Scanner, kind: Token_Kind, loc := #caller_location) {
	text := substring(s.source, s.start, s.current)
	append(
		&s.tokens,
		Token{kind = kind, text = strings.clone(text, s.allocator), line = s.line},
		loc = loc,
	)
}

@(private = "package")
s_is_at_end :: proc(s: ^Scanner) -> bool {
	return s.current >= len(s.source)
}

@(private = "package")
s_match :: proc(s: ^Scanner, expected: string) -> bool {
	if s_is_at_end(s) do return false
	if char_at(s.source, s.current) != expected do return false

	s.current += 1
	return true
}

@(private = "package")
s_peek :: proc(s: ^Scanner) -> string {
	if s_is_at_end(s) do return "\x00"
	return char_at(s.source, s.current)
}

@(private = "package")
s_peek_next :: proc(s: ^Scanner) -> string {
	if s.current + 1 >= len(s.source) do return "\x00"
	return char_at(s.source, s.current + 1)
}

@(private = "package")
s_check_keyword :: proc(s: string) -> Token_Kind {
	switch s {
	case "&&":
		return .AND
	case "||":
		return .OR
	case "if":
		return .IF
	case "else":
		return .ELSE
	case "false":
		return .FALSE
	case "for":
		return .FOR
	case "fn":
		return .FUNCTION
	case "return":
		return .RETURN
	case "true":
		return .TRUE
	case "var":
		return .VAR
	case "let":
		return .LET
	case "const":
		return .CONST
	case "struct":
		return .STRUCT
	}

	return .IDENTIFIER
}

/* 
~Parser 
*/

@(private = "package")
Parser :: struct {
	tokens:    []Token,
	current:   int,
	allocator: mem.Allocator,
	errors:    [dynamic]Any_Err,
}

@(private = "package")
Node :: struct {
	line: int,
}

@(private = "package")
Var_Decl :: struct {
	using node: Node,
	name:       string,
	type:       Type,
	space:      Address_Space,
	access:     Access_Type,
	attributes: []Attribute,
	value:      string,
}

@(private = "package")
Struct_Decl :: struct {
	using node: Node,
	name:       string,
	type:       Type,
	fields:     []Field,
	attributes: []Attribute,
}

@(private = "package")
Function_Decl :: struct {
	using node:   Node,
	name:         string,
	arguments:    []Argument,
	attributes:   []Attribute,
	declarations: []Var_Decl,
	identifiers:  map[string]struct {},
	return_type:  Type,
}

@(private = "package")
Attribute :: struct {
	name:   string,
	values: []string,
}

@(private = "package")
Argument :: struct {
	name: string,
	type: Type,
}

@(private = "package")
Field :: struct {
	name: string,
	type: Type,
}

@(private = "package")
Type :: struct {
	name:   string,
	format: ^Type,
}

@(private = "package")
Address_Space :: enum {
	Private,
	Uniform,
	Storage,
	Workgroup,
	Function,
}

@(private = "package")
Access_Type :: enum {
	Read,
	Write,
	Read_Write,
}

@(private = "package")
Any_Stmt :: union {
	^Var_Decl,
	^Struct_Decl,
	^Function_Decl,
}

@(private = "package")
p_global_decl_or_directive :: proc(p: ^Parser) -> Any_Stmt {
	attrs := p_attribute(p)
	if p_check(p, .VAR) {
		var := p_global_var_decl(p)
		var.attributes = attrs
		return new_clone(var, p.allocator)
	}

	if p_check(p, .STRUCT) {
		decl := p_struct_decl(p)
		decl.attributes = attrs
		return new_clone(decl, p.allocator)
	}

	if p_check(p, .FUNCTION) {
		decl := p_function_decl(p)
		decl.attributes = attrs
		return new_clone(decl, p.allocator)
	}

	p_advance(p)

	return nil
}

@(private = "package")
p_global_var_decl :: proc(p: ^Parser, loc := #caller_location) -> Var_Decl {
	if !p_match(p, .VAR) do return {}
	var_token := p_previous(p)

	decl := Var_Decl {
		line = var_token.line,
	}

	// variable_qualifier: less_than storage_class (comma access_mode)? greater_than
	if p_match(p, .LESS) {
		token := p_consume(p, .IDENTIFIER)
		address_space := token.text
		switch address_space {
		case "storage":
			decl.space = .Storage
		case "uniform":
			decl.space = .Uniform
		case:
			append(
				&p.errors,
				Unexpected_Value_Err {
					msg = fmt.aprintf(
						"Unknown address space: %v",
						address_space,
						allocator = p.allocator,
					),
					line = token.line,
					loc = loc,
				},
			)
		}

		if p_match(p, .COMMA) {
			access_text := p_consume(p, .IDENTIFIER).text
			switch access_text {
			case "read":
				decl.access = .Read
			case "write":
				decl.access = .Write
			case "read_write":
				decl.access = .Read_Write
			}

		}
		p_consume(p, .GREATER)
	}

	decl.name = p_consume(p, .IDENTIFIER).text

	if p_match(p, .COLON) {
		decl.type = p_type_decl(p)
	}

	if p_match(p, .EQUAL) {
		decl.value = p_consume_multi(p, {.NUMBER, .IDENTIFIER}).text
	}

	p_consume(p, .SEMICOLON)

	return decl
}

@(private = "package")
p_type_decl :: proc(p: ^Parser) -> Type {
	p_skip_attributes(p)
	type := Type {
		name = p_consume(p, .IDENTIFIER).text,
	}

	if p_match(p, .LESS) {
		type.format = new_clone(p_type_decl(p), p.allocator)
		if p_match(p, .COMMA) {
			_ = p_consume(p, .IDENTIFIER)
		}
		p_consume(p, .GREATER)
	}

	return type
}

@(private = "package")
p_struct_decl :: proc(p: ^Parser) -> Struct_Decl {
	if !p_match(p, .STRUCT) do return {}
	struct_token := p_previous(p)

	decl := Struct_Decl {
		name = p_consume(p, .IDENTIFIER).text,
		line = struct_token.line,
	}
	fields := make([dynamic]Field, p.allocator)

	p_consume(p, .BRACE_LEFT)
	p_skip_attributes(p) // We skip struct attributes such as @location for now
	for p_check(p, .IDENTIFIER) {
		field_name := p_consume(p, .IDENTIFIER).text
		p_match(p, .COLON)
		field_type := p_type_decl(p)
		append(&fields, Field{name = field_name, type = field_type})
		p_match(p, .COMMA)
		p_skip_attributes(p)
	}
	p_consume(p, .BRACE_RIGHT)

	decl.fields = fields[:]

	return decl
}

@(private = "package")
p_function_decl :: proc(p: ^Parser) -> Function_Decl {
	if !p_match(p, .FUNCTION) do return {}
	fn_token := p_previous(p)

	decl := Function_Decl {
		name = p_consume(p, .IDENTIFIER).text,
		line = fn_token.line,
	}
	args := make([dynamic]Argument, p.allocator)
	declarations := make([dynamic]Var_Decl, p.allocator)

	p_consume(p, .PAREN_LEFT)
	p_attribute(p)
	for p_check(p, .IDENTIFIER) {
		arg_name := p_consume(p, .IDENTIFIER).text
		p_match(p, .COLON)
		arg_type := p_type_decl(p)
		append(&args, Argument{arg_name, arg_type})
		if p_check(p, .COMMA) do p_advance(p)
		p_attribute(p)
	}
	p_consume(p, .PAREN_RIGHT)

	decl.arguments = args[:]

	p_consume(p, .RETURN_TYPE)
	decl.return_type = p_type_decl(p)

	brace_depth := 1
	p_consume(p, .BRACE_LEFT)
	for !p_is_at_end(p) {
		if p_check(p, .BRACE_LEFT) {
			brace_depth += 1
		}
		if p_check(p, .BRACE_RIGHT) {
			brace_depth -= 1
		}

		if brace_depth == 0 {
			break
		}

		p_advance(p)
	}

	return decl
}

@(private = "package")
p_attribute :: proc(p: ^Parser) -> []Attribute {
	attrs := make([dynamic]Attribute, p.allocator)

	for p_match(p, .ATTR) {
		name := p_consume(p, .IDENTIFIER)
		attr := Attribute {
			name = name.text,
		}

		if p_match(p, .PAREN_LEFT) {
			values := make([dynamic]string, p.allocator)
			for p_check_multi(p, {.NUMBER, .IDENTIFIER}) {
				append(&values, p_consume_multi(p, {.NUMBER, .IDENTIFIER}).text)
				p_match(p, .COMMA)
			}

			attr.values = values[:]

			p_consume(p, .PAREN_RIGHT)
		}

		append(&attrs, attr)
	}

	return attrs[:]
}

@(private = "package")
p_skip_attributes :: proc(p: ^Parser) {
	for p_match(p, .ATTR) {
		p_match(p, .IDENTIFIER)
		if p_match(p, .PAREN_LEFT) {
			for p_match(p, .NUMBER, .IDENTIFIER) {
				p_match(p, .COMMA)
			}

			p_consume(p, .PAREN_RIGHT)
		}
	}
}

@(private = "package")
p_consume_multi :: proc(p: ^Parser, kinds: []Token_Kind, loc := #caller_location) -> Token {
	if !p_check_multi(p, kinds) {
		token := p_peek(p)
		append(
			&p.errors,
			Unexpected_Token_Err {
				msg = fmt.aprintf(
					"Expected any of %v, found %v",
					kinds,
					token.kind,
					allocator = p.allocator,
				),
				line = token.line,
				loc = loc,
			},
		)
	}

	return p_advance(p)
}

@(private = "package")
p_consume :: proc(p: ^Parser, kind: Token_Kind, loc := #caller_location) -> Token {
	if !p_check(p, kind) {
		token := p_peek(p)
		append(
			&p.errors,
			Unexpected_Token_Err {
				msg = fmt.aprintf(
					"Expected %v, found %v",
					kind,
					token.kind,
					allocator = p.allocator,
				),
				line = token.line,
				loc = loc,
			},
		)
	}
	return p_advance(p)
}

@(private = "package")
p_match :: proc(p: ^Parser, kinds: ..Token_Kind) -> bool {
	for kind in kinds {
		if p_check(p, kind) {
			p_advance(p)
			return true
		}
	}

	return false
}

@(private = "package")
p_check_multi :: proc(p: ^Parser, kinds: []Token_Kind) -> bool {
	if p_is_at_end(p) do return false
	tk := p_peek(p)
	for kind in kinds {
		if tk.kind == kind do return true
	}
	return false
}

@(private = "package")
p_check :: proc(p: ^Parser, kind: Token_Kind) -> bool {
	if p_is_at_end(p) do return false
	return p_peek(p).kind == kind
}

@(private = "package")
p_advance :: proc(p: ^Parser) -> Token {
	if !p_is_at_end(p) do p.current += 1
	return p_previous(p)
}

@(private = "package")
p_is_at_end :: proc(p: ^Parser) -> bool {
	return p_peek(p).kind == .EOF
}

@(private = "package")
p_peek :: proc(p: ^Parser) -> Token {
	return p.tokens[p.current]
}

@(private = "package")
p_previous :: proc(p: ^Parser) -> Token {
	return p.tokens[p.current - 1]
}


/*
~Reflection
*/

Texture_Type :: struct {
	dimension:    wgpu.TextureViewDimension,
	sample_type:  wgpu.TextureSampleType,
	multisampled: b32,
}

Buffer_Type :: struct {
	type: wgpu.BufferBindingType,
}

Sampler_Type :: struct {
	type: wgpu.SamplerBindingType,
}

Resource_Type :: enum {
	Buffer,
	Texture,
	Sampler,
}

Resource_Info :: struct {
	name:           string,
	type:           Resource_Type,
	group, binding: int,
	visibility:     wgpu.ShaderStageFlags,
	buffer:         wgpu.BufferBindingLayout,
	texture:        wgpu.TextureBindingLayout,
	sampler:        wgpu.SamplerBindingLayout,
}

Function_Info :: struct {
	name:       cstring,
	stage:      wgpu.ShaderStage,
	arguments:  []Argument,
	attributes: []Attribute,
}

Metadata :: struct {
	resources:    map[string]Resource_Info,
	entry_points: []Function_Info,
}

Layout_Entry :: struct {
	group:          int,
	binding:        int,
	visibility:     wgpu.ShaderStageFlags,
	buffer:         wgpu.BufferBindingLayout,
	sampler:        wgpu.SamplerBindingLayout,
	texture:        wgpu.TextureBindingLayout,
	storageTexture: wgpu.StorageTextureBindingLayout,
}

@(private = "package")
r_get_attribute_values :: proc(attrs: []Attribute, name: string) -> ([]string, bool) {
	for attr in attrs {
		if attr.name == name {
			return attr.values, true
		}
	}

	return {}, false
}

@(private = "package")
r_has_attribute :: proc(attrs: []Attribute, name: string) -> bool {
	for attr in attrs {
		if attr.name == name {
			return true
		}
	}

	return false
}

@(private = "package")
r_get_attribute_value_int :: proc(attrs: []Attribute, name: string) -> (int, bool) {
	for attr in attrs {
		if attr.name == name {
			n, ok := strconv.parse_int(attr.values[0], 10)
			return n, ok
		}
	}

	return 0, false
}

@(private = "package")
r_is_entry_point :: proc(f: ^Function_Decl) -> bool {
	for attr in f.attributes {
		switch attr.name {
		case "vertex":
			return true
		case "fragment":
			return true
		case "compute":
			return true
		}
	}

	return false
}

/*
~Utils
*/

@(private = "package")
is_digit :: proc(s: string) -> bool {
	_, ok := strconv.parse_int(s, 10)
	return ok
}

@(private = "package")
is_alpha :: proc(char: string) -> bool {
	return (char >= "a" && char <= "z") || (char >= "A" && char <= "Z") || char == "_"
}

@(private = "package")
is_alpha_numeric :: proc(char: string) -> bool {
	return is_alpha(char) || is_digit(char)
}

@(private = "package")
char_at :: proc(str: string, index: int) -> string {
	return substring(str, index, index + 1)
}

@(private = "package")
substring :: proc(source: string, start: int, end: int, loc := #caller_location) -> string {
	str, ok := strings.substring(source, start, end)
	assert_contextless(ok, "out of bounds", loc)

	return str
}

@(private = "package")
substring_from :: proc(source: string, index: int, loc := #caller_location) -> string {
	str, ok := strings.substring_from(source, index)
	assert_contextless(ok, "out of bounds", loc)

	return str
}

@(private = "package")
substring_to :: proc(source: string, index: int, loc := #caller_location) -> string {
	str, ok := strings.substring_to(source, index)
	assert_contextless(ok, "out of bounds", loc)

	return str
}

/*
~Tests
*/

import "core:testing"

@(private = "file")
shader := `
struct Camera {
	viewProjectionMatrix: mat4x4f,
}
@group(0) @binding(0) @stage(vertex)
var<uniform> camera: Camera;
@group(1) @binding(0) @stage(vertex)
var<storage> transform: array<mat4x4f>;
@group(1) @binding(1) @stage(fragment) var tex: texture_2d<f32>;
@group(1) @binding(2) @stage(fragment) var samp: sampler;

@vertex
fn v_main(@location(0) position: vec3f, @builtin(instance_index) iidx: u32) -> @builtin(position) vec4f {
	return camera.viewProjectionMatrix * transform[iidx] * vec4f(input.position, 1.0);
}

@fragment
fn f_main() -> @location(0) vec4f {
	let texColor = textureSample(tex, samp, vec2f(0, 0));
	return texColor;
}
`


@(test)
test :: proc(t: ^testing.T) {
	metadata, _, _, errs := process(shader, context.temp_allocator)
	defer free_all(context.temp_allocator)

	testing.expect(t, len(errs) == 0)
	vertex_entry_point, _ := get_entry_point(metadata, .Vertex)
	testing.expect(t, vertex_entry_point.name == "v_main")
	fragment_entry_point, _ := get_entry_point(metadata, .Fragment)
	testing.expect(t, fragment_entry_point.name == "f_main")

	{
		testing.expect(t, "camera" in metadata.resources)
		camera := metadata.resources["camera"]
		testing.expect(t, camera.group == 0)
		testing.expect(t, camera.binding == 0)
		testing.expect(t, camera.type == .Buffer)
		testing.expect(t, camera.buffer.type == .Uniform)
		testing.expect(t, camera.visibility == {.Vertex})
	}

	{
		testing.expect(t, "transform" in metadata.resources)
		transform := metadata.resources["transform"]
		testing.expect(t, transform.group == 1)
		testing.expect(t, transform.binding == 0)
		testing.expect(t, transform.type == .Buffer)
		testing.expect(t, transform.buffer.type == .ReadOnlyStorage)
		testing.expect(t, transform.visibility == {.Vertex})
	}

	{
		testing.expect(t, "tex" in metadata.resources)
		resource := metadata.resources["tex"]
		testing.expect(t, resource.type == .Texture)
		testing.expect(t, resource.texture.viewDimension == ._2D)
		testing.expect(t, resource.texture.sampleType == .Float)
		testing.expect(t, resource.texture.multisampled == false)
		testing.expect(t, resource.visibility == {.Fragment})
	}

	{
		testing.expect(t, "samp" in metadata.resources)
		resource := metadata.resources["samp"]
		testing.expect(t, resource.type == .Sampler)
		testing.expect(t, resource.sampler.type == .Filtering)
		testing.expect(t, resource.visibility == {.Fragment})
	}

	layout_0 := generate_layout_desc(metadata, 0, context.temp_allocator)
	layout_1 := generate_layout_desc(metadata, 1, context.temp_allocator)

	testing.expect(t, layout_0.entryCount == 1)
	testing.expect(t, layout_1.entryCount == 3)
}
