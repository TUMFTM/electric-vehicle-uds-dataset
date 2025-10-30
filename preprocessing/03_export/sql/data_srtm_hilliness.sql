select
    ca.city_id,
    stddev(elev.elevation) as elevation_stddev
from
    nextbike.city_areas ca
    -- Start by filtering out cities without bikes/trips
    join (
        select distinct
            bu.city_id
        from
            nextbike.bike_usage bu
    ) as c on c.city_id = ca.city_id
    -- Get NUTS regions for cities
    join nextbike.city_nuts3 cn on ca.city_id = cn.id
    -- Join with population density for city's NUTS region
    join lateral (
        select
            (st_pixelaspolygons (srtm.rast)).geom as bbox,
            (st_pixelaspolygons (srtm.rast)).val as elevation
        from
            elevation.srtm_europe_dgm30 srtm
            join (
                select
                    st_transform (
                        st_buffer (
                            st_transform (am.geom, 3857),
                            10000::double precision
                        ),
                        4326
                    ) as shape
                from
                    administrative_boundaries.nuts_2021_1m am
                where
                    am.nuts_id = cn.nuts_id
            ) b on true
        where
            srtm.rast && b.shape
    ) elev
    -- Filter to elevation bounding boxes within city area
    on st_within (
        elev.bbox,
        st_transform (ca.area, st_srid (elev.bbox))
    )
    -- Aggregate (and order) by city id to compute average elevation
group by
    ca.city_id
order by
    ca.city_id;