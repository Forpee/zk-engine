//! Golden byte-compatibility fixtures.
//!
//! These pin the exact wire (bincode) and transcript (`to_bytes_le`)
//! encodings — and the BN254 legacy challenge derivations — to the byte
//! streams produced by the original `jolt-field` crate, guarding the hard
//! invariant that replacing that crate does not change proof bytes.
//!
//! GENERATED from jolt-field at commit
//! 5b3e39ece1c27586a1f7cc77f24e718cb5d73e10 (branch
//! feat/jolt-field-replacement) by a one-off generator test (deleted in the
//! same change; see this file's history). To regenerate: check out that
//! commit, restore `tests/golden_gen.rs` and the `jolt-field`
//! dev-dependency, run
//! `cargo nextest run -p jolt-field --all-features generate_golden_fixtures`,
//! and splice `target/tmp/golden_fixtures.txt` into the const blocks below.
//!
//! Row format: `(input hex, expected hex)` where the element is
//! `from_bytes_le_reduced(input)` (prime fields) or
//! `from_challenge_bytes` / `from_scalar_challenge_bytes(input)`
//! (challenge fixtures), and `expected` is the canonical LE encoding.
//! Extension rows are `(canonical coefficients, bincode wire hex)`.

#![expect(clippy::unwrap_used, reason = "test code")]
// The whole file is backend fixture data; without a backend there is nothing
// to pin and every item would be dead code under -Dwarnings.

use jolt_field as two;

use two::CanonicalEncoding;

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

/// Element from the reducing decode; canonical bytes and bincode wire must
/// equal the fixture, and the checked decode must round-trip.
fn check_prime_rows<F>(rows: &[(&str, &str)])
where
    F: CanonicalEncoding
        + serde::Serialize
        + serde::de::DeserializeOwned
        + PartialEq
        + std::fmt::Debug
        + Copy,
{
    let cfg = bincode::config::standard();
    for (input, expected) in rows {
        let (input, expected) = (unhex(input), unhex(expected));
        let e = F::from_bytes_le_reduced(&input);
        assert_eq!(e.to_bytes_le_vec(), expected, "transcript bytes diverge");
        let wire = bincode::serde::encode_to_vec(e, cfg).unwrap();
        assert_eq!(wire, expected, "wire bytes diverge");
        assert_eq!(
            F::from_bytes_le_checked(&expected),
            Some(e),
            "checked decode round-trip"
        );
        let (back, read): (F, usize) = bincode::serde::decode_from_slice(&expected, cfg).unwrap();
        assert_eq!((back, read), (e, expected.len()));
    }
}

const FIX_BN254_FR: &[(&str, &str)] = &[
    (
        "0000000000000000000000000000000000000000000000000000000000000000",
        "0000000000000000000000000000000000000000000000000000000000000000",
    ),
    (
        "0100000000000000000000000000000000000000000000000000000000000000",
        "0100000000000000000000000000000000000000000000000000000000000000",
    ),
    (
        "0200000000000000000000000000000000000000000000000000000000000000",
        "0200000000000000000000000000000000000000000000000000000000000000",
    ),
    (
        "000000f093f5e1439170b97948e833285d588181b64550b829a031e1724e6430",
        "000000f093f5e1439170b97948e833285d588181b64550b829a031e1724e6430",
    ),
    (
        "010000f093f5e1439170b97948e833285d588181b64550b829a031e1724e6430",
        "0000000000000000000000000000000000000000000000000000000000000000",
    ),
    (
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "faffff4f1c3496ac29cd609f9576fc362e4679786fa36e662fdf079ac1770a0e",
    ),
    (
        "dae623d2aa29f41845b9a32a1d819bb65e7ca3106f280413e0638df8a0029a4d",
        "d9e623e2163412d5b348eab0d498678e0124228fb8e2b35ab6c35b172eb4351d",
    ),
    (
        "7c41df5c178ab67358fb5375991fe1ba909e7d5a185156624bd8be7123e5d34b",
        "7b41df6c8394d42fc78a9afb5037ad923346fcd8610b06aa21388d90b0966f1b",
    ),
    (
        "f0496a0db0d44aecb71a5457bac1cd1f004ed4e1b8a4ad0fda75893f75f9e34b",
        "ef496a1d1cdf68a826aa9add71d999f7a2f55260025f5d57b0d5575e02ab7f1b",
    ),
    (
        "a3697a40313c6c012a94dd33469644182c96bcbf573df48ae896963ea3675f9e",
        "a0697a70755bc6357642b1c66cdda89f148d383b346c03626bb6019b4a7c320d",
    ),
    (
        "cd49341848ecb614e2aee4b81d74bc679583ce92d4db5b516e29348f760fc986",
        "cb4934382001f38cbfcd71c58ca35417dbd2cb8f6750bbe01ae9d0cc90720026",
    ),
    (
        "4388d11302c44ae8b30ca39efc96d6cf965cba921500e6806331d6ea881d2dcc",
        "3f88d153b2edc2d86e4abdb7daf5062f22fbb48c3be9a49fbcb00f66bde39b0a",
    ),
    (
        "f8cef80001d2e51cdc6e48ab3ae7fb4ef2c2118736b66a9ae1e2a4741d0cc498",
        "f5cef83045f13f51281d1c3e612e60d6dab98d0213e57971640210d1c4209707",
    ),
    (
        "26ebe17a6b629d695def5007330a4a49270921edea58aa1d6891972609397536",
        "25ebe18ad76cbb25cc7e978dea211621cab09f6b34135a653ef1654596ea1006",
    ),
];
const FIX_BN254_FQ: &[(&str, &str)] = &[
    (
        "0000000000000000000000000000000000000000000000000000000000000000",
        "0000000000000000000000000000000000000000000000000000000000000000",
    ),
    (
        "0100000000000000000000000000000000000000000000000000000000000000",
        "0100000000000000000000000000000000000000000000000000000000000000",
    ),
    (
        "0200000000000000000000000000000000000000000000000000000000000000",
        "0200000000000000000000000000000000000000000000000000000000000000",
    ),
    (
        "46fd7cd8168c203c8dca7168916a81975d588181b64550b829a031e1724e6430",
        "46fd7cd8168c203c8dca7168916a81975d588181b64550b829a031e1724e6430",
    ),
    (
        "47fd7cd8168c203c8dca7168916a81975d588181b64550b829a031e1724e6430",
        "0000000000000000000000000000000000000000000000000000000000000000",
    ),
    (
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "9c0d8fc58d435dd33d0bc7f528eb780a2c4679786fa36e662fdf079ac1770a0e",
    ),
    (
        "dae623d2aa29f41845b9a32a1d819bb65e7ca3106f280413e0638df8a0029a4d",
        "93e9a6f9939dd3dcb7ee31c28b161a1f0124228fb8e2b35ab6c35b172eb4351d",
    ),
    (
        "7c41df5c178ab67358fb5375991fe1ba909e7d5a185156624bd8be7123e5d34b",
        "3544628400fe9537cb30e20c08b55f233346fcd8610b06aa21388d90b0966f1b",
    ),
    (
        "f0496a0db0d44aecb71a5457bac1cd1f004ed4e1b8a4ad0fda75893f75f9e34b",
        "a94ced3499482ab02a50e2ee28574c88a2f55260025f5d57b0d5575e02ab7f1b",
    ),
    (
        "a3697a40313c6c012a94dd33469644182c96bcbf573df48ae896963ea3675f9e",
        "ce7103b7ec970a4d823488fa9156c051138d383b346c03626bb6019b4a7c320d",
    ),
    (
        "cd49341848ecb614e2aee4b81d74bc679583ce92d4db5b516e29348f760fc986",
        "3f4f3a671ad4759cc71901e8fa9eb938dad2cb8f6750bbe01ae9d0cc90720026",
    ),
    (
        "4388d11302c44ae8b30ca39efc96d6cf965cba921500e6806331d6ea881d2dcc",
        "2793ddb1a693c8f77ee2dbfcb6ecd07120fbb48c3be9a49fbcb00f66bde39b0a",
    ),
    (
        "f8cef80001d2e51cdc6e48ab3ae7fb4ef2c2118736b66a9ae1e2a4741d0cc498",
        "23d78177bc2d8468340ff37186a77788d9b98d0213e57971640210d1c4209707",
    ),
    (
        "26ebe17a6b629d695def5007330a4a49270921edea58aa1d6891972609397536",
        "dfed64a254d67c2dd024df9ea19fc8b1c9b09f6b34135a653ef1654596ea1006",
    ),
];
const FIX_BN254_FR_CHALLENGE: &[(&str, &str)] = &[
    (
        "00000000000000000000000000000000",
        "0000000000000000000000000000000000000000000000000000000000000000",
    ),
    (
        "ffffffffffffffffffffffffffffffff",
        "922306fba4417702705b58c72aed7f2e72a8da6190063056ab362c65cdc32617",
    ),
    (
        "dae623d2aa29f41845b9a32a1d819bb6",
        "3b6bccb3eae3ebb5d9133924500858a3fac6c42854d0d35ae53b1e8f9f71200e",
    ),
    (
        "5e7ca3106f280413e0638df8a0029a4d",
        "7f37873e17701c776900992bbf4d097271100afd8855cf1859fe2dc56226e00b",
    ),
    (
        "7c41df5c178ab67358fb5375991fe1ba",
        "e77a83d4edd8d4a65ada9fd99b800bf3afba689fbd890e2c3dc10db8948c790f",
    ),
    (
        "909e7d5a185156624bd8be7123e5d34b",
        "3c1f49d192ebb7e7f8a0be53b8e30491a1a98c2e45558f3b7cd443d88538c814",
    ),
    (
        "f0496a0db0d44aecb71a5457bac1cd1f",
        "2be6dfd869df0b3ae480a5ad51a4fc21150cb9ffca8faca48897f93285b78b22",
    ),
    (
        "004ed4e1b8a4ad0fda75893f75f9e34b",
        "f8c2174ea5e97eb646b7b0fd9a6a047dca24236aa1a61e02e964d8aaa051c926",
    ),
];
const FIX_BN254_FR_SCALAR_CHALLENGE: &[(&str, &str)] = &[
    (
        "0000000000000000000000000000000000000000000000000000000000000000",
        "0000000000000000000000000000000000000000000000000000000000000000",
    ),
    (
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "faffff4f1c3496ac29cd609f9576fc362e4679786fa36e662fdf079ac1770a0e",
    ),
    (
        "dae623d2aa29f41845b9a32a1d819bb65e7ca3106f280413e0638df8a0029a4d",
        "499a02e0a8b7dbd0ce414288ee01adbd413a7c17508c78647173632507ea5419",
    ),
    (
        "7c41df5c178ab67358fb5375991fe1ba909e7d5a185156624bd8be7123e5d34b",
        "49d3e54349d314c43f75de24c9ac364000311d9608c85ae81f7627557642791b",
    ),
    (
        "f0496a0db0d44aecb71a5457bac1cd1f004ed4e1b8a4ad0fda75893f75f9e34b",
        "47e3f9b5efb2edcacaeabed1bf337f5faa6bbcb47d3dd9d545ca0d2c4230b82e",
    ),
    (
        "a3697a40313c6c012a94dd33469644182c96bcbf573df48ae896963ea3675f9e",
        "9b5f67d382b5f01cd7a211eae503fbb3003b12c20f0ca401848ba78de78e3c12",
    ),
    (
        "cd49341848ecb614e2aee4b81d74bc679583ce92d4db5b516e29348f760fc986",
        "82c90fb63f5ea15e0c99f5ed702db4f4f25a6f17decd6d016e3526c44cfab70b",
    ),
    (
        "4388d11302c44ae8b30ca39efc96d6cf965cba921500e6806331d6ea881d2dcc",
        "cb2d1d9856e14f1fef75479b49d2286e727e157be85dbcfabeaa9221a0822413",
    ),
];
const FIX_BN254_FQ_CHALLENGE: &[(&str, &str)] = &[
    (
        "00000000000000000000000000000000",
        "0000000000000000000000000000000000000000000000000000000000000000",
    ),
    (
        "ffffffffffffffffffffffffffffffff",
        "00000000000000000000000000000000ffffffffffffffffffffffffffffff1f",
    ),
    (
        "dae623d2aa29f41845b9a32a1d819bb6",
        "00000000000000000000000000000000dae623d2aa29f41845b9a32a1d819b16",
    ),
    (
        "5e7ca3106f280413e0638df8a0029a4d",
        "000000000000000000000000000000005e7ca3106f280413e0638df8a0029a0d",
    ),
    (
        "7c41df5c178ab67358fb5375991fe1ba",
        "000000000000000000000000000000007c41df5c178ab67358fb5375991fe11a",
    ),
    (
        "909e7d5a185156624bd8be7123e5d34b",
        "00000000000000000000000000000000909e7d5a185156624bd8be7123e5d30b",
    ),
    (
        "f0496a0db0d44aecb71a5457bac1cd1f",
        "00000000000000000000000000000000f0496a0db0d44aecb71a5457bac1cd1f",
    ),
    (
        "004ed4e1b8a4ad0fda75893f75f9e34b",
        "00000000000000000000000000000000004ed4e1b8a4ad0fda75893f75f9e30b",
    ),
];
const FIX_BN254_FQ_SCALAR_CHALLENGE: &[(&str, &str)] = &[
    (
        "0000000000000000000000000000000000000000000000000000000000000000",
        "0000000000000000000000000000000000000000000000000000000000000000",
    ),
    (
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "9c0d8fc58d435dd33d0bc7f528eb780a2c4679786fa36e662fdf079ac1770a0e",
    ),
    (
        "dae623d2aa29f41845b9a32a1d819bb65e7ca3106f280413e0638df8a0029a4d",
        "31a50e3e9d5de1efded960cdcaf87600403a7c17508c78647173632507ea5419",
    ),
    (
        "7c41df5c178ab67358fb5375991fe1ba909e7d5a185156624bd8be7123e5d34b",
        "bdd8eb7243a697d347c16d4737a89b61ff301d9608c85ae81f7627557642791b",
    ),
    (
        "f0496a0db0d44aecb71a5457bac1cd1f004ed4e1b8a4ad0fda75893f75f9e34b",
        "2fee0514e458f3e9da82dd169c2a49a2a86bbcb47d3dd9d545ca0d2c4230b82e",
    ),
    (
        "a3697a40313c6c012a94dd33469644182c96bcbf573df48ae896963ea3675f9e",
        "c967f019faf13434e394e81d0b7d1266ff3a12c20f0ca401848ba78de78e3c12",
    ),
    (
        "cd49341848ecb614e2aee4b81d74bc679583ce92d4db5b516e29348f760fc986",
        "6ad41b143404a77d1c3114334d247e37f15a6f17decd6d016e3526c44cfab70b",
    ),
    (
        "4388d11302c44ae8b30ca39efc96d6cf965cba921500e6806331d6ea881d2dcc",
        "8530a0afd34a1127f31b8fac0050dbfe717e157be85dbcfabeaa9221a0822413",
    ),
];

mod bn254 {
    use super::*;

    fn check_challenge_rows<F: CanonicalEncoding>(rows: &[(&str, &str)], scalar: bool) {
        for (input, expected) in rows {
            let (input, expected) = (unhex(input), unhex(expected));
            let e = if scalar {
                F::from_scalar_challenge_bytes(&input)
            } else {
                F::from_challenge_bytes(&input)
            };
            assert_eq!(e.to_bytes_le_vec(), expected, "challenge bytes diverge");
        }
    }

    #[test]
    fn fr_bytes_match_fixtures() {
        check_prime_rows::<two::Fr>(FIX_BN254_FR);
    }

    #[test]
    fn fq_bytes_match_fixtures() {
        check_prime_rows::<two::Fq>(FIX_BN254_FQ);
    }

    #[test]
    fn fr_challenges_match_fixtures() {
        check_challenge_rows::<two::Fr>(FIX_BN254_FR_CHALLENGE, false);
        check_challenge_rows::<two::Fr>(FIX_BN254_FR_SCALAR_CHALLENGE, true);
    }

    #[test]
    fn fq_challenges_match_fixtures() {
        check_challenge_rows::<two::Fq>(FIX_BN254_FQ_CHALLENGE, false);
        check_challenge_rows::<two::Fq>(FIX_BN254_FQ_SCALAR_CHALLENGE, true);
    }
}
