#pragma once

#include <array>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// check if charconv header is available
#if __has_include(<charconv>)
#    define CHESS_USE_CHARCONV
#    include <charconv>
#else
#    include <sstream>
#endif

#include "attacks_fwd.hpp"
#include "color.hpp"
#include "constants.hpp"
#include "coords.hpp"
#include "move.hpp"
#include "movegen_fwd.hpp"
#include "piece.hpp"
#include "utils.hpp"
#include "zobrist.hpp"

namespace chess {

namespace detail {
inline std::optional<int> parseStringViewToInt(std::string_view sv) {
    if (sv.empty()) return std::nullopt;

    std::string_view parsed_sv = sv;
    if (parsed_sv.back() == ';') parsed_sv.remove_suffix(1);

    if (parsed_sv.empty()) return std::nullopt;

#ifdef CHESS_USE_CHARCONV
    int result;
    const char *begin = parsed_sv.data();
    const char *end   = begin + parsed_sv.size();

    auto [ptr, ec] = std::from_chars(begin, end, result);

    if (ec == std::errc() && ptr == end) {
        return result;
    }
#else
    std::string str(parsed_sv);
    std::stringstream ss(str);

    ss.exceptions(std::ios::goodbit);

    int value;
    ss >> value;

    if (!ss.fail() && (ss.eof() || (ss >> std::ws).eof())) {
        return value;
    }
#endif

    return std::nullopt;
}
}  // namespace detail

enum class GameResult { WIN, LOSE, DRAW, NONE };

enum class GameResultReason {
    CHECKMATE,
    STALEMATE,
    INSUFFICIENT_MATERIAL,
    FIFTY_MOVE_RULE,
    THREEFOLD_REPETITION,
    NONE
};

enum class CheckType {
    NO_CHECK,

    // In normal mode, you only get CHECK as a return value when move
    // can actually give a check. No other distinguishing is made.
    CHECK,

    // In detailed mode you get the precise check type of the move.
    // The effort is costly, so only use it when detailed information is needed.
    DIRECT_CHECK = CHECK,
    SINGLE_DISCOVERY_CHECK,
    DIRECT_DISCOVERY_CHECK,  // double check
    DOUBLE_DISCOVERY_CHECK   // double check
};

// A compact representation of the board in 24 bytes,
// does not include the half-move clock or full move number.
using PackedBoard = std::array<std::uint8_t, 24>;

class Board {
    using U64 = std::uint64_t;

   public:
    class CastlingRights {
       public:
        enum class Side : std::uint8_t { KING_SIDE, QUEEN_SIDE };

        constexpr void setCastlingRight(Color color, Side castle, File rook_file) {
            rooks[color][static_cast<int>(castle)] = rook_file;
        }

        constexpr void clear() { rooks[0][0] = rooks[0][1] = rooks[1][0] = rooks[1][1] = File::NO_FILE; }

        constexpr int clear(Color color, Side castle) {
            rooks[color][static_cast<int>(castle)] = File::NO_FILE;
            return color * 2 + static_cast<int>(castle);
        }

        constexpr void clear(Color color) { rooks[color][0] = rooks[color][1] = File::NO_FILE; }

        constexpr bool has(Color color, Side castle) const {
            return rooks[color][static_cast<int>(castle)] != File::NO_FILE;
        }

        constexpr bool has(Color color) const { return has(color, Side::KING_SIDE) || has(color, Side::QUEEN_SIDE); }

        constexpr File getRookFile(Color color, Side castle) const { return rooks[color][static_cast<int>(castle)]; }

        constexpr int hashIndex() const {
            return has(Color::WHITE, Side::KING_SIDE) + 2 * has(Color::WHITE, Side::QUEEN_SIDE) +
                   4 * has(Color::BLACK, Side::KING_SIDE) + 8 * has(Color::BLACK, Side::QUEEN_SIDE);
        }

        constexpr bool isEmpty() const { return !has(Color::WHITE) && !has(Color::BLACK); }

        template <typename T>
        static constexpr Side closestSide(T sq, T pred) {
            return sq > pred ? Side::KING_SIDE : Side::QUEEN_SIDE;
        }

       private:
        std::array<std::array<File, 2>, 2> rooks;
    };

   private:
    struct State {
        U64 key_;
        CastlingRights cr_;
        Square ep_sq_;
        std::uint8_t hfm_;
        Piece captured_;
        std::array<Bitboard, 6> check_sq_;
        Bitboard checkmask_;
        int checks_;
        Bitboard pin_hv_;
        Bitboard pin_d_;
        Bitboard discovery_blocker_;  // own pieces blocking slider attacks to enemy king
    };

    enum class PrivateCtor { CREATE };

    // private constructor to avoid initialization
    Board(PrivateCtor) {}

    [[nodiscard]] void setupChecks() noexcept;

    [[nodiscard]] State &state() noexcept { return states_.back(); }
    [[nodiscard]] const State &cstate() const noexcept { return states_.back(); }

   public:
    explicit Board(std::string_view fen = constants::STARTPOS, bool chess960 = false) {
        states_.reserve(256);
        chess960_ = chess960;
        assert(setFenInternal<true>(constants::STARTPOS));
        setFenInternal<true>(fen);
    }

    static Board fromFen(std::string_view fen) { return Board(fen); }
    static Board fromEpd(std::string_view epd) {
        Board board;
        board.setEpd(epd);
        return board;
    }

    /**
     * @brief Returns true if the given FEN was successfully parsed and set.
     * @param fen
     * @return
     */
    virtual bool setFen(std::string_view fen) { return setFenInternal(fen); }

    /**
     * @brief Returns true if the given EPD was successfully parsed and set.
     * @param epd
     * @return
     */
    bool setEpd(const std::string_view epd) {
        auto parts = utils::splitString(epd, ' ');

        if (parts.size() < 4) return false;

        int hm = 0;
        int fm = 1;

        if (auto it = std::find(parts.begin(), parts.end(), "hmvc"); it != parts.end()) {
            if (std::distance(it, parts.end()) > 1) {
                auto num    = *(it + 1);
                auto parsed = detail::parseStringViewToInt(num);
                if (parsed) hm = *parsed;
            } else {
                return false;
            }
        }

        if (auto it = std::find(parts.begin(), parts.end(), "fmvn"); it != parts.end()) {
            if (std::distance(it, parts.end()) > 1) {
                auto num    = *(it + 1);
                auto parsed = detail::parseStringViewToInt(num);
                if (parsed && *parsed > 0)
                    fm = *parsed;
                else
                    return false;
            } else {
                return false;
            }
        }

        auto fen = std::string(parts[0]) + " " + std::string(parts[1]) + " " + std::string(parts[2]) + " " +
                   std::string(parts[3]) + " " + std::to_string(hm) + " " + std::to_string(fm);

        return setFen(fen);
    }

    /**
     * @brief  Get the current FEN string.
     * @param move_counters
     * @return
     */
    [[nodiscard]] std::string getFen(bool move_counters = true) const {
        std::string ss;
        ss.reserve(100);

        const auto &st = cstate();

        // Loop through the ranks of the board in reverse order
        for (int rank = 7; rank >= 0; rank--) {
            std::uint32_t free_space = 0;

            // Loop through the files of the board
            for (int file = 0; file < 8; file++) {
                // Calculate the square index
                const int sq = rank * 8 + file;

                // If there is a piece at the current square
                if (Piece piece = at(Square(sq)); piece != Piece::NONE) {
                    // If there were any empty squares before this piece,
                    // append the number of empty squares to the FEN string
                    if (free_space) {
                        ss += std::to_string(free_space);
                        free_space = 0;
                    }

                    // Append the character representing the piece to the FEN string
                    ss += static_cast<std::string>(piece);
                } else {
                    // If there is no piece at the current square, increment the
                    // counter for the number of empty squares
                    free_space++;
                }
            }

            // If there are any empty squares at the end of the rank,
            // append the number of empty squares to the FEN string
            if (free_space != 0) {
                ss += std::to_string(free_space);
            }

            // Append a "/" character to the FEN string, unless this is the last rank
            ss += (rank > 0 ? "/" : "");
        }

        // Append " w " or " b " to the FEN string, depending on which player's turn it is
        ss += ' ';
        ss += (stm_ == Color::WHITE ? 'w' : 'b');

        // Append the appropriate characters to the FEN string to indicate
        // whether castling is allowed for each player
        if (st.cr_.isEmpty())
            ss += " -";
        else {
            ss += ' ';
            ss += getCastleString();
        }

        // Append information about the en passant square (if any)
        // and the half-move clock and full move number to the FEN string
        if (st.ep_sq_ == Square::NO_SQ)
            ss += " -";
        else {
            ss += ' ';
            ss += static_cast<std::string>(st.ep_sq_);
        }

        if (move_counters) {
            ss += ' ';
            ss += std::to_string(halfMoveClock());
            ss += ' ';
            ss += std::to_string(fullMoveNumber());
        }

        // Return the resulting FEN string
        return ss;
    }

    [[nodiscard]] std::string getEpd() const {
        std::string ss;
        ss.reserve(100);

        ss += getFen(false);
        ss += " hmvc ";
        ss += std::to_string(halfMoveClock()) + ";";
        ss += " fmvn ";
        ss += std::to_string(fullMoveNumber()) + ";";

        return ss;
    }

    /**
     * @brief Make a move on the board. The move must be legal otherwise the
     * behavior is undefined. EXACT can be set to true to only record
     * the enpassant square if the enemy can legally capture the pawn on their
     * next move.
     * @tparam EXACT
     * @param move
     */
    template <bool EXACT = false>
    void makeMove(const Move move) {
        const auto capture  = at(move.to()) != Piece::NONE && move.typeOf() != Move::CASTLING;
        const auto captured = at(move.to());
        const auto pt       = at<PieceType>(move.from());

        // Validate side to move
        assert((at(move.from()) < Piece::BLACKPAWN) == (stm_ == Color::WHITE));

        auto &newSt       = states_.emplace_back();
        const auto &oldSt = states_[states_.size() - 2];
        newSt.cr_         = oldSt.cr_;
        newSt.hfm_        = oldSt.hfm_ + 1;
        newSt.captured_   = captured;

        plies_++;

        U64 key = oldSt.key_;
        if (oldSt.ep_sq_ != Square::NO_SQ) key ^= Zobrist::enpassant(oldSt.ep_sq_.file());
        newSt.ep_sq_ = Square::NO_SQ;

        if (capture) {
            removePiece(captured, move.to());

            newSt.hfm_ = 0;
            key ^= Zobrist::piece(captured, move.to());

            // remove castling rights if rook is captured
            if (captured.type() == PieceType::ROOK && Rank::back_rank(move.to().rank(), ~stm_)) {
                const auto king_sq = kingSq(~stm_);
                const auto file    = CastlingRights::closestSide(move.to(), king_sq);

                if (newSt.cr_.getRookFile(~stm_, file) == move.to().file()) {
                    key ^= Zobrist::castlingIndex(newSt.cr_.clear(~stm_, file));
                }
            }
        }

        // remove castling rights if king moves
        if (pt == PieceType::KING && newSt.cr_.has(stm_)) {
            key ^= Zobrist::castling(newSt.cr_.hashIndex());
            newSt.cr_.clear(stm_);
            key ^= Zobrist::castling(newSt.cr_.hashIndex());
        } else if (pt == PieceType::ROOK && Square::back_rank(move.from(), stm_)) {
            const auto king_sq = kingSq(stm_);
            const auto file    = CastlingRights::closestSide(move.from(), king_sq);

            // remove castling rights if rook moves from back rank
            if (newSt.cr_.getRookFile(stm_, file) == move.from().file()) {
                key ^= Zobrist::castlingIndex(newSt.cr_.clear(stm_, file));
            }
        } else if (pt == PieceType::PAWN) {
            newSt.hfm_ = 0;

            // double push
            if (Square::value_distance(move.to(), move.from()) == 16) {
                // imaginary attacks from the ep square from the pawn which moved
                Bitboard ep_mask = attacks::pawn(stm_, move.to().ep_square());

                // add enpassant hash if enemy pawns are attacking the square
                if (static_cast<bool>(ep_mask & pieces(PieceType::PAWN, ~stm_))) {
                    int found = -1;

                    // check if the enemy can legally capture the pawn on the next move
                    if constexpr (EXACT) {
                        const auto piece = at(move.from());

                        found = 0;

                        removePieceInternal(piece, move.from());
                        placePieceInternal(piece, move.to());

                        stm_ = ~stm_;

                        bool valid;

                        if (stm_ == Color::WHITE) {
                            valid = movegen::isEpSquareValid<Color::WHITE>(*this, move.to().ep_square());
                        } else {
                            valid = movegen::isEpSquareValid<Color::BLACK>(*this, move.to().ep_square());
                        }

                        if (valid) found = 1;

                        // undo
                        stm_ = ~stm_;

                        removePieceInternal(piece, move.to());
                        placePieceInternal(piece, move.from());
                    }

                    if (found != 0) {
                        assert(at(move.to().ep_square()) == Piece::NONE);
                        newSt.ep_sq_ = move.to().ep_square();
                        key ^= Zobrist::enpassant(move.to().ep_square().file());
                    }
                }
            }
        }

        if (move.typeOf() == Move::CASTLING) {
            assert(at<PieceType>(move.from()) == PieceType::KING);
            assert(at<PieceType>(move.to()) == PieceType::ROOK);

            const bool king_side = move.to() > move.from();
            const auto rookTo    = Square::castling_rook_square(king_side, stm_);
            const auto kingTo    = Square::castling_king_square(king_side, stm_);

            const auto king = at(move.from());
            const auto rook = at(move.to());

            removePiece(king, move.from());
            removePiece(rook, move.to());

            assert(king == Piece(PieceType::KING, stm_));
            assert(rook == Piece(PieceType::ROOK, stm_));

            placePiece(king, kingTo);
            placePiece(rook, rookTo);

            key ^= Zobrist::piece(king, move.from()) ^ Zobrist::piece(king, kingTo);
            key ^= Zobrist::piece(rook, move.to()) ^ Zobrist::piece(rook, rookTo);
        } else if (move.typeOf() == Move::PROMOTION) {
            const auto piece_pawn = Piece(PieceType::PAWN, stm_);
            const auto piece_prom = Piece(move.promotionType(), stm_);

            removePiece(piece_pawn, move.from());
            placePiece(piece_prom, move.to());

            key ^= Zobrist::piece(piece_pawn, move.from()) ^ Zobrist::piece(piece_prom, move.to());
        } else {
            assert(at(move.from()) != Piece::NONE);
            assert(at(move.to()) == Piece::NONE);

            const auto piece = at(move.from());

            removePiece(piece, move.from());
            placePiece(piece, move.to());

            key ^= Zobrist::piece(piece, move.from()) ^ Zobrist::piece(piece, move.to());
            if (move.typeOf() == Move::ENPASSANT) {
                assert(at<PieceType>(move.to().ep_square()) == PieceType::PAWN);

                const auto piece = Piece(PieceType::PAWN, ~stm_);

                removePiece(piece, move.to().ep_square());

                key ^= Zobrist::piece(piece, move.to().ep_square());
            }
        }

        key ^= Zobrist::sideToMove();
        stm_ = ~stm_;

        newSt.key_ = key;

        setupChecks();
    }

    void unmakeMove(const Move move) noexcept {
        const auto &prev = states_.back();

        stm_ = ~stm_;
        plies_--;

        if (move.typeOf() == Move::CASTLING) {
            const bool king_side    = move.to() > move.from();
            const auto rook_from_sq = Square(king_side ? File::FILE_F : File::FILE_D, move.from().rank());
            const auto king_to_sq   = Square(king_side ? File::FILE_G : File::FILE_C, move.from().rank());

            assert(at<PieceType>(rook_from_sq) == PieceType::ROOK);
            assert(at<PieceType>(king_to_sq) == PieceType::KING);

            const auto rook = at(rook_from_sq);
            const auto king = at(king_to_sq);

            removePiece(rook, rook_from_sq);
            removePiece(king, king_to_sq);

            assert(king == Piece(PieceType::KING, stm_));
            assert(rook == Piece(PieceType::ROOK, stm_));

            placePiece(king, move.from());
            placePiece(rook, move.to());

        } else if (move.typeOf() == Move::PROMOTION) {
            const auto pawn  = Piece(PieceType::PAWN, stm_);
            const auto piece = at(move.to());

            assert(piece.type() == move.promotionType());
            assert(piece.type() != PieceType::PAWN);
            assert(piece.type() != PieceType::KING);
            assert(piece.type() != PieceType::NONE);

            removePiece(piece, move.to());
            placePiece(pawn, move.from());

            if (prev.captured_ != Piece::NONE) {
                assert(at(move.to()) == Piece::NONE);
                placePiece(prev.captured_, move.to());
            }

        } else {
            assert(at(move.to()) != Piece::NONE);
            assert(at(move.from()) == Piece::NONE);

            const auto piece = at(move.to());

            removePiece(piece, move.to());
            placePiece(piece, move.from());

            if (move.typeOf() == Move::ENPASSANT) {
                const auto pawn   = Piece(PieceType::PAWN, ~stm_);
                //const auto pawnTo = static_cast<Square>((&cstate() - 1)->ep_sq_ ^ 8);
                const auto pawnTo = static_cast<Square>((&cstate() - 1)->ep_sq_.ep_square());

                assert(at(pawnTo) == Piece::NONE);

                placePiece(pawn, pawnTo);
            } else if (prev.captured_ != Piece::NONE) {
                assert(at(move.to()) == Piece::NONE);

                placePiece(prev.captured_, move.to());
            }
        }

        states_.pop_back();
    }

    /**
     * @brief Make a null move. (Switches the side to move)
     */
    void makeNullMove() {
        auto &newSt       = states_.emplace_back();
        const auto &oldSt = states_[states_.size() - 2];

        U64 key = oldSt.key_ ^ Zobrist::sideToMove();
        if (oldSt.ep_sq_ != Square::NO_SQ) key ^= Zobrist::enpassant(oldSt.ep_sq_.file());
        newSt.key_      = key;
        newSt.cr_       = oldSt.cr_;
        newSt.ep_sq_    = Square::NO_SQ;
        newSt.captured_ = Piece::NONE;

        stm_ = ~stm_;

        plies_++;

        setupChecks();
    }

    /**
     * @brief Unmake a null move. (Switches the side to move)
     */
    void unmakeNullMove() noexcept {
        plies_--;

        stm_ = ~stm_;

        states_.pop_back();
    }

    /**
     * @brief Get the occupancy bitboard for the color.
     * @param color
     * @return
     */
    [[nodiscard]] Bitboard us(Color color) const noexcept { return occ_bb_[color]; }

    /**
     * @brief Get the occupancy bitboard for the opposite color.
     * @param color
     * @return
     */
    [[nodiscard]] Bitboard them(Color color) const noexcept { return us(~color); }

    /**
     * @brief Get the occupancy bitboard for both colors.
     * Faster than calling all() or us(Color::WHITE) | us(Color::BLACK).
     * @return
     */
    [[nodiscard]] Bitboard occ() const noexcept { return occ_bb_[0] | occ_bb_[1]; }

    /**
     * @brief Get the occupancy bitboard for all pieces, should be only used internally.
     * @return
     */
    [[nodiscard]] Bitboard all() const noexcept { return us(Color::WHITE) | us(Color::BLACK); }

    /**
     * @brief Returns the square of the king for a certain color
     * @param color
     * @return
     */
    [[nodiscard]] Square kingSq(Color color) const noexcept {
        assert(pieces(PieceType::KING, color) != 0ull);
        return pieces(PieceType::KING, color).lsb();
    }

    /**
     * @brief Returns all pieces of a certain type and color
     * @param type
     * @param color
     * @return
     */
    [[nodiscard]] Bitboard pieces(PieceType type, Color color) const noexcept {
        return pieces_bb_[type] & occ_bb_[color];
    }

    /**
     * @brief Returns all pieces of a certain type
     * @param type
     * @return
     */
    [[nodiscard]] Bitboard pieces(PieceType type) const noexcept { return pieces_bb_[type]; }

    template <typename... Pieces, typename = std::enable_if_t<(std::is_convertible_v<Pieces, PieceType> && ...)>>
    [[nodiscard]] Bitboard pieces(Pieces... pieces) const noexcept {
        return (pieces_bb_[static_cast<PieceType>(pieces)] | ...);
    }

    /**
     * @brief Returns either the piece or the piece type on a square
     * @tparam T
     * @param sq
     * @return
     */
    template <typename T = Piece>
    [[nodiscard]] T at(Square sq) const noexcept {
        assert(sq.is_valid());

        if constexpr (std::is_same_v<T, PieceType>) {
            return board_[sq.index()].type();
        } else {
            return board_[sq.index()];
        }
    }

    /**
     * @brief Checks if a move is a capture, enpassant moves are also considered captures.
     * @param move
     * @return
     */
    bool isCapture(const Move move) const noexcept {
        return (at(move.to()) != Piece::NONE && move.typeOf() != Move::CASTLING) || move.typeOf() == Move::ENPASSANT;
    }

    /**
     * @brief Get the current zobrist hash key of the board
     * @return
     */
    [[nodiscard]] U64 hash() const noexcept { return cstate().key_; }
    [[nodiscard]] Color sideToMove() const noexcept { return stm_; }
    [[nodiscard]] Square enpassantSq() const noexcept { return cstate().ep_sq_; }
    [[nodiscard]] CastlingRights castlingRights() const noexcept { return cstate().cr_; }
    [[nodiscard]] std::uint32_t halfMoveClock() const noexcept { return cstate().hfm_; }
    [[nodiscard]] std::uint32_t fullMoveNumber() const noexcept { return 1 + plies_ / 2; }

    /**
     * @brief Get the position of all enemy sliders that pin our pieces to our king
     * @return
     */
    [[nodiscard]] Bitboard pinner() const noexcept { return (cstate().pin_hv_ | cstate().pin_d_) & us(~stm_); }

    /**
     * @brief Get the position our pieces that are pinned to our king
     * @return
     */
    [[nodiscard]] Bitboard pinned() const noexcept { return (cstate().pin_hv_ | cstate().pin_d_) & us(stm_); }

    void set960(bool is960) {
        chess960_ = is960;
        if (!original_fen_.empty()) setFen(original_fen_);
    }

    /**
     * @brief Checks if the current position is a chess960, aka. FRC/DFRC position.
     * @return
     */
    [[nodiscard]] bool chess960() const noexcept { return chess960_; }

    /**
     * @brief Get the castling rights as a string
     * @return
     */
    [[nodiscard]] std::string getCastleString() const {
        static const auto get_file = [](const CastlingRights &cr, Color c, CastlingRights::Side side) {
            auto file = static_cast<std::string>(cr.getRookFile(c, side));
            return c == Color::WHITE ? std::toupper(file[0]) : file[0];
        };

        const auto &st = cstate();

        if (chess960_) {
            std::string ss;

            for (auto color : {Color::WHITE, Color::BLACK})
                for (auto side : {CastlingRights::Side::KING_SIDE, CastlingRights::Side::QUEEN_SIDE})
                    if (st.cr_.has(color, side)) ss += get_file(st.cr_, color, side);

            return ss;
        }

        std::string ss;

        if (st.cr_.has(Color::WHITE, CastlingRights::Side::KING_SIDE)) ss += 'K';
        if (st.cr_.has(Color::WHITE, CastlingRights::Side::QUEEN_SIDE)) ss += 'Q';
        if (st.cr_.has(Color::BLACK, CastlingRights::Side::KING_SIDE)) ss += 'k';
        if (st.cr_.has(Color::BLACK, CastlingRights::Side::QUEEN_SIDE)) ss += 'q';

        return ss;
    }

    /**
     * @brief Checks if the current position is a repetition, set this to 1 if
     * you are writing a chess engine.
     * @param count
     * @return
     */
    [[nodiscard]] bool isRepetition(int count = 2) const noexcept {
        std::uint8_t c = 0;

        // We start the loop from the back and go forward in moves, at most to the
        // last move which reset the half-move counter because repetitions cant
        // be across half-moves.
        const int size = static_cast<int>(states_.size());
        const int hfm  = halfMoveClock();
        const auto key = hash();

        for (int i = size - 3; i >= 0 && i >= size - hfm - 1; i -= 2) {
            if (states_[i].key_ == key) c++;
            if (c == count) return true;
        }

        return false;
    }

    /**
     * @brief Checks if the current position is a draw by 50 move rule.
     * Keep in mind that by the rules of chess, if the position has 50 half
     * moves it's not necessarily a draw, since checkmate has higher priority,
     * call getHalfMoveDrawType,
     * to determine whether the position is a draw or checkmate.
     * @return
     */
    [[nodiscard]] bool isHalfMoveDraw() const noexcept { return halfMoveClock() >= 100; }

    /**
     * @brief Only call this function if isHalfMoveDraw() returns true.
     * @return
     */
    [[nodiscard]] std::pair<GameResultReason, GameResult> getHalfMoveDrawType() const noexcept {
        Movelist movelist;
        movegen::legalmoves(movelist, *this);

        if (movelist.empty() && inCheck()) {
            return {GameResultReason::CHECKMATE, GameResult::LOSE};
        }

        return {GameResultReason::FIFTY_MOVE_RULE, GameResult::DRAW};
    }

    /**
     * @brief Basic check if the current position is a draw by insufficient material.
     * @return
     */
    [[nodiscard]] bool isInsufficientMaterial() const noexcept {
        const auto count = occ().count();

        // only kings, draw
        if (count == 2) return true;

        // only bishop + knight, cant mate
        if (count == 3) {
            if (pieces(PieceType::BISHOP, Color::WHITE) || pieces(PieceType::BISHOP, Color::BLACK)) return true;
            if (pieces(PieceType::KNIGHT, Color::WHITE) || pieces(PieceType::KNIGHT, Color::BLACK)) return true;
        }

        // same colored bishops, cant mate
        if (count == 4) {
            if (pieces(PieceType::BISHOP, Color::WHITE) && pieces(PieceType::BISHOP, Color::BLACK) &&
                Square::same_color(pieces(PieceType::BISHOP, Color::WHITE).lsb(),
                                   pieces(PieceType::BISHOP, Color::BLACK).lsb()))
                return true;

            // one side with two bishops which have the same color
            auto white_bishops = pieces(PieceType::BISHOP, Color::WHITE);
            auto black_bishops = pieces(PieceType::BISHOP, Color::BLACK);

            if (white_bishops.count() == 2) {
                if (Square::same_color(white_bishops.lsb(), white_bishops.msb())) return true;
            } else if (black_bishops.count() == 2) {
                if (Square::same_color(black_bishops.lsb(), black_bishops.msb())) return true;
            }
        }

        return false;
    }

    /**
     * @brief Checks if the game is over. Returns GameResultReason::NONE if the game is not over.
     * This function calculates all legal moves for the current position to check if the game is over.
     * If you are writing a chess engine you should not use this function.
     * @return
     */
    [[nodiscard]] std::pair<GameResultReason, GameResult> isGameOver() const noexcept {
        if (isHalfMoveDraw()) return getHalfMoveDrawType();
        if (isInsufficientMaterial()) return {GameResultReason::INSUFFICIENT_MATERIAL, GameResult::DRAW};
        if (isRepetition()) return {GameResultReason::THREEFOLD_REPETITION, GameResult::DRAW};

        Movelist movelist;
        movegen::legalmoves(movelist, *this);

        if (movelist.empty()) {
            if (inCheck()) return {GameResultReason::CHECKMATE, GameResult::LOSE};
            return {GameResultReason::STALEMATE, GameResult::DRAW};
        }

        return {GameResultReason::NONE, GameResult::NONE};
    }

    /**
     * @brief Checks if a square is attacked by the given color.
     * @param square
     * @param color
     * @return
     */
    [[nodiscard]] bool isAttacked(Square square, Color color) const noexcept {
        // cheap checks first
        if (attacks::pawn(~color, square) & pieces(PieceType::PAWN, color)) return true;
        if (attacks::knight(square) & pieces(PieceType::KNIGHT, color)) return true;
        if (attacks::king(square) & pieces(PieceType::KING, color)) return true;
        if (attacks::bishop(square, occ()) & pieces(PieceType::BISHOP, PieceType::QUEEN) & us(color)) return true;
        if (attacks::rook(square, occ()) & pieces(PieceType::ROOK, PieceType::QUEEN) & us(color)) return true;

        return false;
    }

    /**
     * @brief Checks if the current side to move is in check
     * @return
     */
    [[nodiscard]] bool inCheck() const noexcept { return cstate().checks_ > 0; }


    /**
     * @brief Checks if the current side to move is in a double check
     * @return
     */
    [[nodiscard]] bool inDoubleCheck() const noexcept { return cstate().checks_ == 2; }

    /**
     * @brief Returns the position of all enemy pieces that check our king 
     * @return
     */
    [[nodiscard]] Bitboard checkers() const noexcept {
        return attacks::attackers(*this, ~stm_, kingSq(stm_)) & us(~stm_);
    }

    /**
     * @brief Test if a move can deliver a check
     * @param move
     * @tparam Detail -> enable if you like to have a detailed check type
     * @return the type of a check
     */
    template <bool Detail = false>
    [[nodiscard]] CheckType givesCheck(const Move &move) const noexcept;

    /**
     * @brief Checks if the given color has at least 1 piece thats not pawn and not king
     * @param color
     * @return
     */
    [[nodiscard]] bool hasNonPawnMaterial(Color color) const noexcept {
        return bool(us(color) ^ (pieces(PieceType::PAWN, PieceType::KING) & us(color)));
    }

    /**
     * @brief Calculates the zobrist hash key of the board, expensive! Prefer using hash().
     * @return
     */
    [[nodiscard]] U64 zobrist() const noexcept {
        U64 key = 0ULL;

        auto pieces = occ();
        auto &st    = cstate();

        while (pieces) {
            const Square sq = pieces.pop();
            key ^= Zobrist::piece(at(sq), sq);
        }

        if (st.ep_sq_ != Square::NO_SQ) key ^= Zobrist::enpassant(st.ep_sq_.file());

        if (stm_ == Color::WHITE) key ^= Zobrist::sideToMove();

        key ^= Zobrist::castling(st.cr_.hashIndex());

        return key;
    }

    [[nodiscard]] Bitboard getCastlingPath(Color c, bool isKingSide) const noexcept {
        return castling_path[c][isKingSide];
    }

    friend std::ostream &operator<<(std::ostream &os, const Board &board);

    friend class movegen;

    /**
     * @brief Compresses the board into a PackedBoard.
     */
    class Compact {
        friend class Board;
        Compact() = default;

       public:
        /**
         * @brief Compresses the board into a PackedBoard
         * @param board
         * @return
         */
        static PackedBoard encode(const Board &board) { return encodeState(board); }

        static PackedBoard encode(std::string_view fen, bool chess960 = false) { return encodeState(fen, chess960); }

        /**
         * @brief Creates a Board object from a PackedBoard
         * @param compressed
         * @param chess960 If the board is a chess960 position, set this to true
         * @return
         */
        static Board decode(const PackedBoard &compressed, bool chess960 = false) {
            Board board     = Board(PrivateCtor::CREATE);
            board.chess960_ = chess960;
            decode(board, compressed);
            return board;
        }

       private:
        /**
         * A compact board representation can be achieved in 24 bytes,
         * we use 8 bytes (64bit) to store the occupancy bitboard,
         * and 16 bytes (128bit) to store the pieces (plus some special information).
         *
         * Each of the 16 bytes can store 2 pieces, since chess only has 12 different pieces,
         * we can represent the pieces from 0 to 11 in 4 bits (a nibble) and use the other 4 bit for
         * the next piece.
         * Since we need to store information about enpassant, castling rights and the side to move,
         * we can use the remaining 4 bits to store this information.
         *
         * However we need to store the information and the piece information together.
         * This means in our case that
         * 12 -> enpassant + a pawn, we can deduce the color of the pawn from the rank of the square
         * 13 -> white rook with castling rights, we later use the file to deduce if it's a short or long castle
         * 14 -> black rook with castling rights, we later use the file to deduce if it's a short or long castle
         * 15 -> black king and black is side to move
         *
         * We will later deduce the square of the pieces from the occupancy bitboard.
         */
        static PackedBoard encodeState(const Board &board) {
            PackedBoard packed{};

            packed[0] = board.occ().getBits() >> 56;
            packed[1] = (board.occ().getBits() >> 48) & 0xFF;
            packed[2] = (board.occ().getBits() >> 40) & 0xFF;
            packed[3] = (board.occ().getBits() >> 32) & 0xFF;
            packed[4] = (board.occ().getBits() >> 24) & 0xFF;
            packed[5] = (board.occ().getBits() >> 16) & 0xFF;
            packed[6] = (board.occ().getBits() >> 8) & 0xFF;
            packed[7] = board.occ().getBits() & 0xFF;

            auto offset = 8 * 2;
            auto occ    = board.occ();

            const auto &st = board.cstate();

            while (occ) {
                // we now fill the packed array, since our convertedpiece only actually needs 4 bits,
                // we can store 2 pieces in one byte.
                const auto sq      = Square(occ.pop());
                const auto shift   = (offset % 2 == 0 ? 4 : 0);
                const auto meaning = convertMeaning(st.cr_, board.sideToMove(), st.ep_sq_, sq, board.at(sq));
                const auto nibble  = meaning << shift;

                packed[offset / 2] |= nibble;
                offset++;
            }

            return packed;
        }

        static PackedBoard encodeState(std::string_view fen, bool chess960 = false) {
            // fallback to slower method
            if (chess960) {
                return encodeState(Board(fen, true));
            }

            PackedBoard packed{};

            while (fen[0] == ' ') fen.remove_prefix(1);

            const auto params     = split_string_view<6>(fen);
            const auto position   = params[0].has_value() ? *params[0] : "";
            const auto move_right = params[1].has_value() ? *params[1] : "w";
            const auto castling   = params[2].has_value() ? *params[2] : "-";
            const auto en_passant = params[3].has_value() ? *params[3] : "-";

            const auto ep  = en_passant == "-" ? Square::NO_SQ : Square(en_passant);
            const auto stm = (move_right == "w") ? Color::WHITE : Color::BLACK;

            CastlingRights cr;

            for (char i : castling) {
                if (i == '-') break;

                const auto king_side  = CastlingRights::Side::KING_SIDE;
                const auto queen_side = CastlingRights::Side::QUEEN_SIDE;

                if (i == 'K') cr.setCastlingRight(Color::WHITE, king_side, File::FILE_H);
                if (i == 'Q') cr.setCastlingRight(Color::WHITE, queen_side, File::FILE_A);
                if (i == 'k') cr.setCastlingRight(Color::BLACK, king_side, File::FILE_H);
                if (i == 'q') cr.setCastlingRight(Color::BLACK, queen_side, File::FILE_A);

                assert(i == 'K' || i == 'Q' || i == 'k' || i == 'q');

                continue;
            }

            const auto parts = split_string_view<8>(position, '/');

            int offset   = 8 * 2;
            int square   = 0;
            Bitboard occ = 0ull;

            for (auto i = parts.rbegin(); i != parts.rend(); i++) {
                auto part = *i;

                for (char curr : *part) {
                    if (isdigit(curr)) {
                        square += (curr - '0');
                    } else if (curr == '/') {
                        square++;
                    } else {
                        const auto p     = Piece(std::string_view(&curr, 1));
                        const auto shift = (offset % 2 == 0 ? 4 : 0);

                        packed[offset / 2] |= convertMeaning(cr, stm, ep, Square(square), p) << shift;
                        offset++;

                        occ.set(square);
                        ++square;
                    }
                }
            }

            packed[0] = occ.getBits() >> 56;
            packed[1] = (occ.getBits() >> 48) & 0xFF;
            packed[2] = (occ.getBits() >> 40) & 0xFF;
            packed[3] = (occ.getBits() >> 32) & 0xFF;
            packed[4] = (occ.getBits() >> 24) & 0xFF;
            packed[5] = (occ.getBits() >> 16) & 0xFF;
            packed[6] = (occ.getBits() >> 8) & 0xFF;
            packed[7] = occ.getBits() & 0xFF;

            return packed;
        }

        static void decode(Board &board, const PackedBoard &compressed) {
            Bitboard occupied = 0ull;

            for (int i = 0; i < 8; i++) {
                occupied |= Bitboard(compressed[i]) << (56 - i * 8);
            }

            int offset           = 16;
            int white_castle_idx = 0, black_castle_idx = 0;
            File white_castle[2] = {File::NO_FILE, File::NO_FILE};
            File black_castle[2] = {File::NO_FILE, File::NO_FILE};

            // clear board state

            board.pieces_bb_.fill(0ULL);
            board.occ_bb_.fill(0ULL);
            board.board_.fill(Piece::NONE);
            board.plies_ = 0;

            board.stm_ = Color::WHITE;

            board.states_.clear();
            auto &st = board.states_.emplace_back();
            st.cr_.clear();
            st.hfm_ = 0;

            board.original_fen_.clear();

            // place pieces back on the board
            while (occupied) {
                const auto sq     = Square(occupied.pop());
                const auto nibble = compressed[offset / 2] >> (offset % 2 == 0 ? 4 : 0) & 0b1111;
                const auto piece  = convertPiece(nibble);

                if (piece != Piece::NONE) {
                    board.placePiece(piece, sq);

                    offset++;
                    continue;
                }

                // Piece has a special meaning, interpret it from the raw integer
                // pawn with ep square behind it
                if (nibble == 12) {
                    st.ep_sq_ = sq.ep_square();
                    // depending on the rank this is a white or black pawn
                    auto color = sq.rank() == Rank::RANK_4 ? Color::WHITE : Color::BLACK;
                    board.placePiece(Piece(PieceType::PAWN, color), sq);
                }
                // castling rights for white
                else if (nibble == 13) {
                    assert(white_castle_idx < 2);
                    white_castle[white_castle_idx++] = sq.file();
                    board.placePiece(Piece(PieceType::ROOK, Color::WHITE), sq);
                }
                // castling rights for black
                else if (nibble == 14) {
                    assert(black_castle_idx < 2);
                    black_castle[black_castle_idx++] = sq.file();
                    board.placePiece(Piece(PieceType::ROOK, Color::BLACK), sq);
                }
                // black to move
                else if (nibble == 15) {
                    board.stm_ = Color::BLACK;
                    board.placePiece(Piece(PieceType::KING, Color::BLACK), sq);
                }

                offset++;
            }

            // reapply castling
            for (int i = 0; i < 2; i++) {
                if (white_castle[i] != File::NO_FILE) {
                    const auto king_sq = board.kingSq(Color::WHITE);
                    const auto file    = white_castle[i];
                    const auto side    = CastlingRights::closestSide(file, king_sq.file());

                    st.cr_.setCastlingRight(Color::WHITE, side, file);
                }

                if (black_castle[i] != File::NO_FILE) {
                    const auto king_sq = board.kingSq(Color::BLACK);
                    const auto file    = black_castle[i];
                    const auto side    = CastlingRights::closestSide(file, king_sq.file());

                    st.cr_.setCastlingRight(Color::BLACK, side, file);
                }
            }

            if (board.stm_ == Color::BLACK) {
                board.plies_++;
            }

            st.key_ = board.zobrist();

            board.setupChecks();
        }

        // 1:1 mapping of Piece::internal() to the compressed piece
        static std::uint8_t convertPiece(Piece piece) { return static_cast<int>(piece.internal()); }

        // for pieces with a special meaning return Piece::NONE since this is otherwise not used
        static Piece convertPiece(std::uint8_t piece) {
            if (piece >= 12) return Piece::NONE;
            return Piece(Piece::underlying(piece));
        }

        // 12 => theres an ep square behind the pawn, rank will be deduced from the rank
        // 13 => any white rook with castling rights, side will be deduced from the file
        // 14 => any black rook with castling rights, side will be deduced from the file
        // 15 => black king and black is side to move
        static std::uint8_t convertMeaning(const CastlingRights &cr, Color stm, Square ep, Square sq, Piece piece) {
            if (piece.type() == PieceType::PAWN && ep != Square::NO_SQ) {
                if (Square(sq.index() ^ 8) == ep) return 12;
            }

            if (piece.type() == PieceType::ROOK) {
                if (piece.color() == Color::WHITE && Square::back_rank(sq, Color::WHITE) &&
                    (cr.getRookFile(Color::WHITE, CastlingRights::Side::KING_SIDE) == sq.file() ||
                     cr.getRookFile(Color::WHITE, CastlingRights::Side::QUEEN_SIDE) == sq.file()))
                    return 13;
                if (piece.color() == Color::BLACK && Square::back_rank(sq, Color::BLACK) &&
                    (cr.getRookFile(Color::BLACK, CastlingRights::Side::KING_SIDE) == sq.file() ||
                     cr.getRookFile(Color::BLACK, CastlingRights::Side::QUEEN_SIDE) == sq.file()))
                    return 14;
            }

            if (piece.type() == PieceType::KING && piece.color() == Color::BLACK && stm == Color::BLACK) {
                return 15;
            }

            return convertPiece(piece);
        }
    };

   protected:
    virtual void placePiece(Piece piece, Square sq) { placePieceInternal(piece, sq); }

    virtual void removePiece(Piece piece, Square sq) { removePieceInternal(piece, sq); }

    std::vector<State> states_;

    std::array<Bitboard, 6> pieces_bb_ = {};
    std::array<Bitboard, 2> occ_bb_    = {};
    std::array<Piece, 64> board_       = {};

    std::uint16_t plies_ = 0;
    Color stm_           = Color::WHITE;

    bool chess960_ = false;

    std::array<std::array<Bitboard, 2>, 2> castling_path = {};

   private:
    void removePieceInternal(Piece piece, Square sq) {
        assert(board_[sq.index()] == piece && piece != Piece::NONE);

        auto type  = piece.type();
        auto color = piece.color();
        auto index = sq.index();

        assert(type != PieceType::NONE);
        assert(color != Color::NONE);
        assert(index >= 0 && index < 64);

        pieces_bb_[type].clear(index);
        occ_bb_[color].clear(index);
        board_[index] = Piece::NONE;
    }

    void placePieceInternal(Piece piece, Square sq) {
        assert(board_[sq.index()] == Piece::NONE);

        auto type  = piece.type();
        auto color = piece.color();
        auto index = sq.index();

        assert(type != PieceType::NONE);
        assert(color != Color::NONE);
        assert(index >= 0 && index < 64);

        pieces_bb_[type].set(index);
        occ_bb_[color].set(index);
        board_[index] = piece;
    }

    template <bool ctor = false>
    bool setFenInternal(std::string_view fen) {
        original_fen_ = fen;

        reset();

        while (!fen.empty() && fen[0] == ' ') fen.remove_prefix(1);

        if (fen.empty()) return false;

        const auto params     = split_string_view<6>(fen);
        const auto position   = params[0].has_value() ? *params[0] : "";
        const auto move_right = params[1].has_value() ? *params[1] : "w";
        const auto castling   = params[2].has_value() ? *params[2] : "-";
        const auto en_passant = params[3].has_value() ? *params[3] : "-";
        const auto half_move  = params[4].has_value() ? *params[4] : "0";
        const auto full_move  = params[5].has_value() ? *params[5] : "1";

        if (position.empty()) return false;

        if (move_right != "w" && move_right != "b") return false;

        auto &st = state();

        const auto half_move_opt = detail::parseStringViewToInt(half_move).value_or(0);
        st.hfm_                  = half_move_opt;

        const auto full_move_opt = detail::parseStringViewToInt(full_move).value_or(1);
        plies_                   = full_move_opt;

        plies_ = plies_ * 2 - 2;

        if (en_passant != "-") {
            if (!Square::is_valid_string_sq(en_passant)) {
                return false;
            }

            st.ep_sq_ = Square(en_passant);
            if (st.ep_sq_ == Square::NO_SQ) return false;
        }

        stm_     = (move_right == "w") ? Color::WHITE : Color::BLACK;

        if (stm_ == Color::BLACK) plies_++;

        auto square = 56;
        for (char curr : position) {
            if (isdigit(curr)) {
                square += (curr - '0');
            } else if (curr == '/') {
                square -= 16;
            } else {
                auto p = Piece(std::string_view(&curr, 1));
                if (p == Piece::NONE || !Square::is_valid_sq(square) || at(square) != Piece::NONE) return false;

                if constexpr (ctor) {
                    placePieceInternal(p, Square(square));
                } else {
                    placePiece(p, square);
                }

                ++square;
            }
        }

        static const auto find_rook = [](const Board &board, CastlingRights::Side side, Color color) -> File {
            const auto king_side = CastlingRights::Side::KING_SIDE;
            const auto king_sq   = board.kingSq(color);
            const auto sq_corner = Square(side == king_side ? Square::SQ_H1 : Square::SQ_A1).relative_square(color);

            const auto start = side == king_side ? king_sq + 1 : king_sq - 1;

            for (Square sq = start; (side == king_side ? sq <= sq_corner : sq >= sq_corner);
                 (side == king_side ? sq++ : sq--)) {
                if (board.at<PieceType>(sq) == PieceType::ROOK && board.at(sq).color() == color) {
                    return sq.file();
                }
            }

            return File(File::NO_FILE);
        };

        // Parse castling rights
        for (char i : castling) {
            if (i == '-') break;

            const auto king_side  = CastlingRights::Side::KING_SIDE;
            const auto queen_side = CastlingRights::Side::QUEEN_SIDE;

            if (!chess960_) {
                if (i == 'K')
                    st.cr_.setCastlingRight(Color::WHITE, king_side, File::FILE_H);
                else if (i == 'Q')
                    st.cr_.setCastlingRight(Color::WHITE, queen_side, File::FILE_A);
                else if (i == 'k')
                    st.cr_.setCastlingRight(Color::BLACK, king_side, File::FILE_H);
                else if (i == 'q')
                    st.cr_.setCastlingRight(Color::BLACK, queen_side, File::FILE_A);
                else
                    return false;

                continue;
            }

            // chess960 castling detection
            const auto color   = isupper(i) ? Color::WHITE : Color::BLACK;
            const auto king_sq = kingSq(color);

            if (i == 'K' || i == 'k') {
                auto file = find_rook(*this, king_side, color);
                if (file == File::NO_FILE) return false;
                st.cr_.setCastlingRight(color, king_side, file);
            } else if (i == 'Q' || i == 'q') {
                auto file = find_rook(*this, queen_side, color);
                if (file == File::NO_FILE) return false;
                st.cr_.setCastlingRight(color, queen_side, file);
            } else {
                const auto file = File(std::string_view(&i, 1));
                if (file == File::NO_FILE) return false;
                const auto side = CastlingRights::closestSide(file, king_sq.file());
                st.cr_.setCastlingRight(color, side, file);
            }
        }

        setupChecks();

        if (st.ep_sq_ != Square::NO_SQ && !((st.ep_sq_.rank() == Rank::RANK_3 && stm_ == Color::BLACK) ||
                                            (st.ep_sq_.rank() == Rank::RANK_6 && stm_ == Color::WHITE))) {
            st.ep_sq_ = Square::NO_SQ;
        }

        if (st.ep_sq_ != Square::NO_SQ) {
            bool valid;

            if (stm_ == Color::WHITE) {
                valid = movegen::isEpSquareValid<Color::WHITE>(*this, st.ep_sq_);
            } else {
                valid = movegen::isEpSquareValid<Color::BLACK>(*this, st.ep_sq_);
            }

            if (!valid) st.ep_sq_ = Square::NO_SQ;
        }

        st.key_ = zobrist();

        // init castling_path
        for (Color c : {Color::WHITE, Color::BLACK}) {
            const auto king_from = kingSq(c);

            for (const auto side : {CastlingRights::Side::KING_SIDE, CastlingRights::Side::QUEEN_SIDE}) {
                if (!st.cr_.has(c, side)) continue;

                const auto rook_from = Square(st.cr_.getRookFile(c, side), king_from.rank());
                const auto king_to   = Square::castling_king_square(side == CastlingRights::Side::KING_SIDE, c);
                const auto rook_to   = Square::castling_rook_square(side == CastlingRights::Side::KING_SIDE, c);

                castling_path[c][side == CastlingRights::Side::KING_SIDE] =
                    (movegen::between(rook_from, rook_to) | movegen::between(king_from, king_to)) &
                    ~(Bitboard::fromSquare(king_from) | Bitboard::fromSquare(rook_from));
            }
        }

        return true;
    }

    template <int N>
    std::array<std::optional<std::string_view>, N> static split_string_view(std::string_view fen,
                                                                            char delimiter = ' ') {
        std::array<std::optional<std::string_view>, N> arr = {};

        std::size_t start = 0;
        std::size_t end   = 0;

        for (std::size_t i = 0; i < N; i++) {
            end = fen.find(delimiter, start);
            if (end == std::string::npos) {
                arr[i] = fen.substr(start);
                break;
            }
            arr[i] = fen.substr(start, end - start);
            start  = end + 1;
        }

        return arr;
    }

    void reset() {
        pieces_bb_.fill(0ULL);
        occ_bb_.fill(0ULL);
        board_.fill(Piece::NONE);

        stm_   = Color::WHITE;
        plies_ = 1;
        states_.clear();
        auto &st = states_.emplace_back();
        st.key_  = 0ull;
        st.cr_.clear();
        st.ep_sq_ = Square::NO_SQ;
        st.hfm_   = 0;
    }

    [[nodiscard]] inline Bitboard getSniper(Bitboard occupied) const noexcept {
        const auto ksq    = kingSq(~stm_);
        const auto us_occ = us(stm_);
        const auto bishop = attacks::bishop(ksq, occupied) & pieces(PieceType::BISHOP, PieceType::QUEEN) & us_occ;
        const auto rook   = attacks::rook(ksq, occupied) & pieces(PieceType::ROOK, PieceType::QUEEN) & us_occ;
        return bishop | rook;
    }

// store the original fen string
    // useful when setting up a frc position and the user called set960(true) afterwards
    std::string original_fen_;
};

inline std::ostream &operator<<(std::ostream &os, const Board &b) {
    for (int i = 63; i >= 0; i -= 8) {
        for (int j = 7; j >= 0; j--) {
            os << " " << static_cast<std::string>(b.board_[i - j]);
        }

        os << "\n";
    }

    os << "\n\nFEN:  " << b.getFen();
    os << "\nHash: " << std::hex << b.hash() << std::dec << std::endl;

    os << std::endl;

    return os;
}

template <bool Detail>
inline CheckType Board::givesCheck(const Move &move) const noexcept {
    assert(at(move.from()).color() == stm_);

    const auto from   = move.from();
    const auto to     = move.to();
    const auto ksq    = kingSq(~stm_);
    const auto toBB   = Bitboard::fromSquare(to);
    const auto fromBB = Bitboard::fromSquare(from);
    const auto oc     = occ() ^ fromBB;

    auto direct_check = [&, this]() { return bool(cstate().check_sq_[at<PieceType>(from)] & toBB); };

    auto discovery_check = [&, this]() {
        return bool(!(movegen::line(from, to) & pieces(PieceType::KING, ~stm_))) || move.typeOf() == Move::CASTLING;
    };

    auto ep_check = [&, this]() {
        auto epBB = Bitboard::fromSquare(enpassantSq().ep_square());
        return getSniper((oc ^ epBB) | toBB);
    };

    if constexpr (Detail) {

        // Check if the moving piece is a blocker.
        if (cstate().discovery_blocker_ & fromBB) {

            // Blocker is moving away from the attack line.
            if (discovery_check()) {

                // We can also directly attack the king.
                if (direct_check()) return CheckType::DIRECT_DISCOVERY_CHECK;

                // Tricky: This is THE most rare checking move in chess
                // when an enpassant reveals two snipers at once.
                if (move.typeOf() == Move::ENPASSANT) {
#if __cpp_lib_int_pow2 >= 202002L
                    if (!std::has_single_bit(ep_check().getBits()))
#else
                    if (ep_check().count() == 2)
#endif
                        return CheckType::DOUBLE_DISCOVERY_CHECK;

                    // Only one slider involved.
                    return CheckType::SINGLE_DISCOVERY_CHECK;
                }

                // The moving piece moves diagonally and checks the king.
                return CheckType::SINGLE_DISCOVERY_CHECK;
            }

            // King is on its second home rank.
            // There is a slider behind our pawn,
            // it can capture enpassant the pawn and attack the king.
            else if (move.typeOf() == Move::ENPASSANT && direct_check())
                return CheckType::DIRECT_CHECK;
            return CheckType::NO_CHECK;
        }

        if (direct_check()) return CheckType::DIRECT_CHECK;

    } else {
        if (direct_check()) return CheckType::CHECK;

        if (cstate().discovery_blocker_ & fromBB)
            return discovery_check() ? CheckType::CHECK : CheckType::NO_CHECK;
    }

    switch (move.typeOf()) {
        case Move::NORMAL:
            return CheckType::NO_CHECK;

        case Move::PROMOTION: {
            Bitboard attacks;

            switch (move.promotionType()) {
                case static_cast<int>(PieceType::KNIGHT):
                    attacks = attacks::knight(to);
                    break;
                case static_cast<int>(PieceType::BISHOP):
                    attacks = attacks::bishop(to, oc);
                    break;
                case static_cast<int>(PieceType::ROOK):
                    attacks = attacks::rook(to, oc);
                    break;
                default:
                    attacks = attacks::queen(to, oc);
            }

            if (attacks & pieces(PieceType::KING, ~stm_)) return (Detail) ? CheckType::DIRECT_CHECK : CheckType::CHECK;
            return CheckType::NO_CHECK;
        }

        case Move::ENPASSANT: {
            if (ep_check()) return (Detail) ? CheckType::SINGLE_DISCOVERY_CHECK : CheckType::CHECK;
            return CheckType::NO_CHECK;
        }

        case Move::CASTLING: {
            auto rookBB = Bitboard::fromSquare(Square::castling_rook_square(to > from, stm_));
            if (attacks::rook(ksq, occ()) & rookBB)
                return (Detail) ? CheckType::SINGLE_DISCOVERY_CHECK : CheckType::CHECK;
            return CheckType::NO_CHECK;
        }
    }

    assert(false);
    return CheckType::NO_CHECK;  // Prevent a compiler warning
}

[[nodiscard]] inline void Board::setupChecks() noexcept {
    auto &st = state();
    auto ksq = kingSq(~stm_);

    // Check squares
    st.check_sq_[0] = attacks::pawn(~stm_, ksq);
    st.check_sq_[1] = attacks::knight(ksq);
    st.check_sq_[2] = attacks::bishop(ksq, occ());
    st.check_sq_[3] = attacks::rook(ksq, occ());
    st.check_sq_[4] = st.check_sq_[2] | st.check_sq_[3];
    st.check_sq_[5] = 0;

    // Discovery blockers
    const auto occ_us  = us(stm_);
    const auto occ_opp = us(~stm_);

    auto snipers      = getSniper(0);
    const auto occ_sn = occ() ^ snipers;

    st.discovery_blocker_ = 0ull;

    while (snipers) {
        const auto blocker = movegen::between(ksq, snipers.pop()) & occ_sn;
        if (blocker.count() == 1) st.discovery_blocker_ |= blocker & occ_us;
    }

    // Check and pin masks
    ksq = kingSq(stm_);

    if (stm_ == Color::WHITE) {
        const auto c  = movegen::checkMask<Color::WHITE>(*this, ksq);
        st.checkmask_ = c.first;
        st.checks_    = c.second;
        st.pin_hv_    = movegen::pinMask<Color::WHITE, PieceType::ROOK>(*this, ksq, occ_opp, occ_us);
        st.pin_d_     = movegen::pinMask<Color::WHITE, PieceType::BISHOP>(*this, ksq, occ_opp, occ_us);
    } else {
        const auto c  = movegen::checkMask<Color::BLACK>(*this, ksq);
        st.checkmask_ = c.first;
        st.checks_    = c.second;
        st.pin_hv_    = movegen::pinMask<Color::BLACK, PieceType::ROOK>(*this, ksq, occ_opp, occ_us);
        st.pin_d_     = movegen::pinMask<Color::BLACK, PieceType::BISHOP>(*this, ksq, occ_opp, occ_us);
    }

    assert(st.checks_ <= 2);
}

}  // namespace  chess
