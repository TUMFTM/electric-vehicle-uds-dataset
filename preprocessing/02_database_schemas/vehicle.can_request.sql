-- vehicle.can_request definition

-- Drop table

-- DROP TABLE vehicle.can_request;

CREATE TABLE vehicle.can_request (
	request_id int8 DEFAULT nextval('vehicle.can_request_seq'::regclass) NOT NULL,
	can_type vehicle.can_type NULL,
	value_id int8 NULL,
	group_id int2 NULL,
	can_id int8 NULL,
	reg_0 int2 NULL,
	reg_1 int2 NULL,
	reg_2 int2 NULL,
	reg_3 int2 NULL,
	convtype text NULL,
	"add" float8 NULL,
	mult float8 NULL,
	div int4 NULL,
	def_rate int2 NULL,
	description text NULL,
	byteorder text NULL,
	params text NULL,
	start_bit int4 NULL,
	"bit_length" int4 NULL,
	mux_byte int4 NULL,
	mux_value int4 NULL,
	CONSTRAINT obd_request_key1 PRIMARY KEY (request_id),
	CONSTRAINT can_request_fk FOREIGN KEY (value_id) REFERENCES vehicle.can_value(value_id)
);