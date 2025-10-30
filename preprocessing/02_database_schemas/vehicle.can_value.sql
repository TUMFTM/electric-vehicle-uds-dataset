-- vehicle.can_value definition

-- Drop table

-- DROP TABLE vehicle.can_value;

CREATE TABLE vehicle.can_value (
	value_id int8 DEFAULT nextval('vehicle.can_value_id_seq'::regclass) NOT NULL,
	name_de text NULL,
	name_en text NULL,
	variable_name text NULL,
	unit text NULL,
	min_val float4 NULL,
	max_val float4 NULL,
	CONSTRAINT can_value_id_key PRIMARY KEY (value_id)
);