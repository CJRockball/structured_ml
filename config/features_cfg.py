from pydantic import BaseModel, field_validator
from .global_cfg import global_cfg


class FeatureConfig(BaseModel):
    target:      str
    cats:        list[str]
    nums:        list[str]
    target_type: str

    model_config = {"frozen": True}

    @field_validator("cats", "nums")
    @classmethod
    def no_empty_lists(cls, v):
        assert len(v) >= 0, "Feature list cannot be negative"
        return v

    @field_validator("nums", mode="after")
    @classmethod
    def no_overlap_with_cats(cls, v, info):
        cats = info.data.get("cats", [])
        overlap = set(v) & set(cats)
        assert not overlap, f"Features appear in both cats and nums: {overlap}"
        return v

    @property
    def all_features(self) -> list[str]:
        """All non-target features in a stable order: cats first, then nums."""
        return self.cats + self.nums



features = FeatureConfig(
    target = 'class',
    cats   = ['spectral_type', 'galaxy_population'],
    nums   = ['alpha', 'delta','u','g','r','i','z','redshift'],
    target_type = 'str'
)

