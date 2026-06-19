import numpy as np

from arrayscope.display.shader_mapping import (
    ShaderComponent,
    ShaderMapping,
    ShaderScale,
    TexturePlaneKind,
    apply_phase_lut,
    apply_scale,
    cpu_display_rgba,
    default_gray_lut,
    extract_component,
    mapped_scalar,
    normalize_lut_rgb,
    pack_texture_data,
    phase_lut_indices,
    shader_component_uniform,
    shader_mapping_with_lut,
    window_intensity,
)


def test_component_extraction_matches_numpy_for_complex_values():
    data = np.array([1 + 2j, -3 - 4j, np.nan + 1j], dtype=np.complex64)

    np.testing.assert_allclose(extract_component(data, ShaderComponent.REAL), np.real(data))
    np.testing.assert_allclose(extract_component(data, ShaderComponent.IMAG), np.imag(data))
    np.testing.assert_allclose(extract_component(data, ShaderComponent.ABS), np.abs(data))
    np.testing.assert_allclose(extract_component(data, ShaderComponent.ANGLE), np.angle(data))


def test_shader_component_uniform_matches_component_order():
    assert shader_component_uniform(ShaderComponent.REAL) == 0.0
    assert shader_component_uniform("imag") == 1.0
    assert shader_component_uniform(ShaderComponent.ABS) == 2.0
    assert shader_component_uniform(ShaderComponent.ANGLE) == 3.0
    assert shader_component_uniform(ShaderComponent.COMPLEX_PHASE) == 3.0


def test_scale_oracle_handles_linear_log_symlog_and_nonfinite_values():
    data = np.array([-10.0, -1.0, 0.0, 1.0, 10.0, np.inf, np.nan], dtype=np.float32)

    np.testing.assert_array_equal(apply_scale(data, ShaderScale.LINEAR), data)
    np.testing.assert_allclose(apply_scale(data, ShaderScale.SYMLOG), np.sign(data) * np.log10(1.0 + np.abs(data)))
    log = apply_scale(data, ShaderScale.LOG)
    assert np.isneginf(log[0])
    assert np.isneginf(log[2])
    assert np.isinf(log[5])
    assert np.isnan(log[6])


def test_mapping_combines_component_and_scale():
    data = np.array([1 + 1j, 10 + 0j], dtype=np.complex64)
    mapping = ShaderMapping(component=ShaderComponent.ABS, scale=ShaderScale.LOG)

    np.testing.assert_allclose(mapped_scalar(data, mapping), np.log10(np.abs(data)))


def test_window_intensity_clips_and_maps_nonfinite_values():
    values = np.array([-1.0, 0.0, 0.5, 1.0, 2.0, np.nan, np.inf, -np.inf], dtype=np.float32)

    np.testing.assert_allclose(
        window_intensity(values, (0.0, 1.0)),
        np.array([0.0, 0.0, 0.5, 1.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32),
    )


def test_phase_lut_indices_are_deterministic_at_key_angles():
    data = np.array([1 + 0j, 1j, -1 + 0j], dtype=np.complex64)

    np.testing.assert_array_equal(phase_lut_indices(data, 5), np.array([2, 3, 4]))


def test_phase_lut_sampling_interpolates_colormap_like_gpu_shader():
    lut = np.array([[0, 0, 0], [255, 0, 0]], dtype=np.uint8)

    color, magnitude = apply_phase_lut(np.array([1 + 0j], dtype=np.complex64), lut)

    np.testing.assert_array_equal(color, np.array([[128, 0, 0]], dtype=np.uint8))
    np.testing.assert_allclose(magnitude, np.array([1.0], dtype=np.float32))


def test_rg32f_pack_preserves_real_and_imag_float32_values():
    data = np.array([[1 + 2j, -3 - 4j]], dtype=np.complex64)

    packed = pack_texture_data(data, TexturePlaneKind.COMPLEX_RG32F)

    assert packed.dtype == np.float32
    assert packed.shape == (1, 2, 2)
    np.testing.assert_array_equal(packed[..., 0], np.real(data).astype(np.float32))
    np.testing.assert_array_equal(packed[..., 1], np.imag(data).astype(np.float32))


def test_cpu_display_rgba_uses_alpha_zero_for_nan_scalar():
    data = np.array([[0.0, 1.0, np.nan]], dtype=np.float32)
    mapping = ShaderMapping(component=ShaderComponent.REAL, scale=ShaderScale.LINEAR, levels=(0.0, 1.0))

    rgba = cpu_display_rgba(data, mapping)

    assert rgba.shape == (1, 3, 4)
    assert rgba[0, 0, 0] == 0
    assert rgba[0, 1, 0] == 255
    assert rgba[0, 2, 3] == 0


def test_phase_color_cpu_oracle_windows_scaled_magnitude():
    data = np.array([[1 + 0j, 10 + 0j, 100 + 0j]], dtype=np.complex64)
    lut = np.full((4, 3), 255, dtype=np.uint8)
    mapping = ShaderMapping(
        component=ShaderComponent.ABS,
        scale=ShaderScale.LOG,
        levels=(0.0, 2.0),
        lut_data=lut,
        display_mode="phase_color",
    )

    rgba = cpu_display_rgba(data, mapping)

    np.testing.assert_array_equal(rgba[0, :, 0], np.array([0, 127, 255], dtype=np.uint8))


def test_phase_color_cpu_oracle_maps_angle_through_lut_levels_without_intensity_window():
    data = np.array([[-1j, 1 + 0j, 1j]], dtype=np.complex64)
    lut = np.array(
        [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
        ],
        dtype=np.uint8,
    )
    mapping = ShaderMapping(
        component=ShaderComponent.ANGLE,
        levels=(-np.pi / 2.0, np.pi / 2.0),
        lut_data=lut,
        display_mode="phase_color",
    )

    rgba = cpu_display_rgba(data, mapping)

    np.testing.assert_array_equal(rgba[0, :, :3], lut)
    np.testing.assert_array_equal(rgba[0, :, 3], np.array([255, 255, 255], dtype=np.uint8))


def test_default_gray_lut_and_float_rgba_normalization_are_canonical_uint8_rgb():
    gray = default_gray_lut(3)
    np.testing.assert_array_equal(
        gray,
        np.array([[0, 0, 0], [128, 128, 128], [255, 255, 255]], dtype=np.uint8),
    )

    normalized = normalize_lut_rgb(
        np.array([[0.0, 0.5, 1.0, 0.25], [1.0, 0.0, 0.5, 1.0]], dtype=np.float32)
    )

    assert normalized.flags.c_contiguous
    assert normalized.dtype == np.uint8
    np.testing.assert_array_equal(
        normalized,
        np.array([[0, 127, 255], [255, 0, 127]], dtype=np.uint8),
    )


def test_shader_mapping_with_lut_preserves_semantics_and_replaces_only_lut_state():
    base = ShaderMapping(
        component=ShaderComponent.IMAG,
        scale=ShaderScale.LOG,
        levels=(2.0, 8.0),
        histogram_source_policy="semantic",
    )
    lut = np.array([[0, 0, 255], [255, 0, 0]], dtype=np.uint8)

    mapped = shader_mapping_with_lut(base, lut, lut_identity=("display-lut", 7))

    assert mapped.component == base.component
    assert mapped.scale == base.scale
    assert mapped.levels == base.levels
    assert mapped.histogram_source_policy == base.histogram_source_policy
    assert mapped.lut_identity == ("display-lut", 7)
    np.testing.assert_array_equal(mapped.lut_data, lut)
    assert base.lut_data is None


def test_scalar_cpu_oracle_uses_the_frame_lut_instead_of_implicit_grayscale():
    data = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    lut = np.array([[0, 0, 255], [255, 0, 0]], dtype=np.uint8)
    mapping = ShaderMapping(levels=(0.0, 1.0), lut_data=lut)

    rgba = cpu_display_rgba(data, mapping)

    np.testing.assert_array_equal(
        rgba[0, :, :3],
        np.array([[0, 0, 255], [128, 0, 128], [255, 0, 0]], dtype=np.uint8),
    )
    np.testing.assert_array_equal(rgba[0, :, 3], np.array([255, 255, 255], dtype=np.uint8))
